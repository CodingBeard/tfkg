package data

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/codingbeard/cberrors"
	"github.com/codingbeard/cblog"
	"github.com/codingbeard/cbutil"
	"github.com/codingbeard/tfkg/preprocessor"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"github.com/remeh/sizedwaitgroup"
	"io"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type SingleFileDataset struct {
	ClassCounts  map[int]int
	ClassWeights map[int]float32
	Count        int

	classCountsLock     *sync.Mutex
	filePath            string
	filePool            *sync.Pool
	concurrentFileLimit int32
	openFileCount       *int32
	ignoreParseErrors   bool
	skipHeaders         bool
	shuffled            bool
	generatorOffset     *int32
	generatorOffsetLock *sync.Mutex
	cacheDir            string
	categoryOffset      int
	columnProcessors    []*preprocessor.Processor
	lineOffsets         []int64
	trainPercent        float32
	valPercent          float32
	testPercent         float32
	trainCount          int32
	valCount            int32
	testCount           int32
	mode                GeneratorMode
	offset              int32
	limit               int32
	filter              func(line []string) bool
	categoryTokenizer   *preprocessor.Tokenizer
	maxRowsForFit       int

	logger       *cblog.Logger
	errorHandler *cberrors.ErrorsContainer
}

type SingleFileDatasetConfig struct {
	FilePath               string
	CacheDir               string
	CategoryOffset         int
	TrainPercent           float32
	ValPercent             float32
	TestPercent            float32
	IgnoreParseErrors      bool
	SkipHeaders            bool
	RowFilter              func(line []string) bool
	ConcurrentFileLimit    int32
	MaxRowsForProcessorFit int
	ClassWeights           map[int]float32
}

func NewSingleFileDataset(
	logger *cblog.Logger,
	errorHandler *cberrors.ErrorsContainer,
	config SingleFileDatasetConfig,
	columnProcessors ...*preprocessor.Processor,
) (*SingleFileDataset, error) {

	logger.InfoF("data", "Initialising single file dataset at: %s", config.FilePath)

	_, e := os.Stat(config.FilePath)
	if e != nil {
		errorHandler.Error(e)
		return nil, e
	}
	if config.ConcurrentFileLimit == 0 {
		config.ConcurrentFileLimit = 1
	}
	if config.MaxRowsForProcessorFit == 0 {
		config.MaxRowsForProcessorFit = 1000000
	}

	if config.ClassWeights == nil {
		config.ClassWeights = make(map[int]float32)
	}

	var openFileCount int32
	var generatorOffset int32

	d := &SingleFileDataset{
		logger:              logger,
		errorHandler:        errorHandler,
		classCountsLock:     &sync.Mutex{},
		filePath:            config.FilePath,
		skipHeaders:         config.SkipHeaders,
		ignoreParseErrors:   config.IgnoreParseErrors,
		cacheDir:            config.CacheDir,
		categoryOffset:      config.CategoryOffset,
		columnProcessors:    columnProcessors,
		trainPercent:        config.TrainPercent,
		valPercent:          config.ValPercent,
		testPercent:         config.TestPercent,
		ClassCounts:         make(map[int]int),
		ClassWeights:        config.ClassWeights,
		filter:              config.RowFilter,
		concurrentFileLimit: config.ConcurrentFileLimit,
		openFileCount:       &openFileCount,
		generatorOffset:     &generatorOffset,
		generatorOffsetLock: &sync.Mutex{},
		maxRowsForFit:       config.MaxRowsForProcessorFit,
	}

	d.filePool = &sync.Pool{
		New: func() interface{} {
			currentFileCount := atomic.LoadInt32(d.openFileCount)
			if currentFileCount >= d.concurrentFileLimit {
				return nil
			}

			file, e := os.Open(d.filePath)
			if e != nil {
				d.errorHandler.Error(e)
			}

			return file
		},
	}

	e = os.MkdirAll(config.CacheDir, os.ModePerm)
	if e != nil && e != os.ErrExist {
		errorHandler.Error(e)
		return nil, e
	}

	if _, e := os.Stat(filepath.Join(config.CacheDir, "category-tokenizer.json")); e == nil {
		d.categoryTokenizer = preprocessor.NewTokenizer(
			errorHandler,
			1,
			-1,
			preprocessor.TokenizerConfig{IsCategoryTokenizer: true, DisableFiltering: true},
		)
		e = d.categoryTokenizer.Load(filepath.Join(config.CacheDir, "category-tokenizer.json"))
		if e != nil {
			errorHandler.Error(e)
			return nil, e
		}
	}

	e = d.readLineOffsets()
	if e != nil {
		return nil, e
	}

	d.trainCount = int32(math.Ceil(float64(len(d.lineOffsets)) * float64(d.trainPercent)))
	d.valCount = int32(math.Ceil(float64(len(d.lineOffsets)) * float64(d.valPercent)))
	d.testCount = int32(math.Ceil(float64(len(d.lineOffsets)) * float64(d.testPercent)))

	e = d.fitPreProcessors()
	if e != nil {
		return nil, e
	}

	return d, nil
}

type fileStatsCache struct {
	LineOffsets  []int64
	Count        int
	ClassCounts  map[int]int
	ClassWeights map[int]float32
}

func (d *SingleFileDataset) readLineOffsets() error {
	cacheFileName := "file-stats.json"
	cacheFileBytes, e := ioutil.ReadFile(filepath.Join(d.cacheDir, cacheFileName))
	if e != nil && !errors.Is(e, os.ErrNotExist) {
		d.errorHandler.Error(e)
		return e
	} else if e == nil {
		var cache fileStatsCache
		e = json.Unmarshal(cacheFileBytes, &cache)
		if e != nil {
			d.errorHandler.Error(e)
			return e
		}
		d.logger.InfoF("data", "Loading line offsets and stats from cache file")

		d.lineOffsets = cache.LineOffsets
		d.Count = cache.Count
		d.ClassCounts = cache.ClassCounts
		if len(d.ClassWeights) == 0 {
			d.ClassWeights = cache.ClassWeights
		}

		d.logger.InfoF("data", "Found %d rows. Got class counts: %#v Got class weights: %#v", d.Count, d.ClassCounts, d.ClassWeights)

		return nil
	}

	d.logger.InfoF("data", "Reading line offsets and counting stats")

	file := d.filePool.Get().(*os.File)

	reader := bufio.NewReader(file)

	offset := int64(0)

	lastPrint := time.Now().Unix()
	progress, lastProgress := 0, 0
	skippedHeaders, zeroAdded := false, false
	swg := sizedwaitgroup.New(128)
	var errs []error
	for true {
		readBytes, e := reader.ReadBytes('\n')

		if errors.Is(e, io.EOF) {
			break
		}
		offset += int64(len(readBytes))
		if !skippedHeaders && d.skipHeaders {
			skippedHeaders = true
			continue
		} else if !d.skipHeaders && !zeroAdded {
			zeroAdded = true
			d.lineOffsets = append(d.lineOffsets, 0)
		}

		if len(errs) > 0 {
			for _, e := range errs {
				return e
			}
		}

		swg.Add()
		go func(readBytes []byte, offset int64) {
			defer swg.Done()
			csvReader := csv.NewReader(bytes.NewBuffer(readBytes))
			line, e := csvReader.Read()
			if e != nil && !errors.Is(e, io.EOF) {
				if d.ignoreParseErrors {
					return
				}
				d.errorHandler.Error(e)
				errs = append(errs, e)
				return
			}
			if len(line) == 0 {
				return
			}
			if d.filter != nil {
				if !d.filter(line) {
					return
				}
			}
			if len(line) < d.categoryOffset {
				if d.ignoreParseErrors {
					return
				}
				e = fmt.Errorf("len(line) (%d) < d.categoryOffset (%d)", len(line), d.categoryOffset)
				d.errorHandler.Error(e)
				errs = append(errs, e)
				return
			}

			d.lineOffsets = append(d.lineOffsets, offset)
			d.Count++

			category := line[d.categoryOffset]

			categoryInt, e := strconv.Atoi(category)
			// TODO: this magical behaviour could be nicer
			if e != nil {
				if d.categoryTokenizer == nil {
					d.categoryTokenizer = preprocessor.NewTokenizer(
						d.errorHandler,
						1,
						-1,
						preprocessor.TokenizerConfig{IsCategoryTokenizer: true, DisableFiltering: true},
					)
				}
				d.categoryTokenizer.Fit(category)
				categoryInt = int(d.categoryTokenizer.Tokenize(category)[0])
			}

			d.classCountsLock.Lock()
			count := d.ClassCounts[categoryInt]
			count++
			d.ClassCounts[categoryInt] = count
			d.classCountsLock.Unlock()
		}(readBytes, offset)

		now := time.Now().Unix()
		if now > lastPrint {
			lastPrint = now
			fmt.Print(fmt.Sprintf("\rReading offsets and counting: %d %d/s", progress, progress-lastProgress))
			lastProgress = progress
		}
		progress++
	}
	swg.Wait()
	fmt.Println()

	if len(d.ClassWeights) == 0 {
		majorClassCount := 0
		for _, count := range d.ClassCounts {
			if count > majorClassCount {
				majorClassCount = count
			}
		}
		for class, count := range d.ClassCounts {
			d.ClassWeights[class] = float32(majorClassCount) / float32(count)
		}
	}

	cacheBytes, e := json.Marshal(fileStatsCache{
		LineOffsets:  d.lineOffsets,
		Count:        d.Count,
		ClassCounts:  d.ClassCounts,
		ClassWeights: d.ClassWeights,
	})
	if e != nil {
		d.errorHandler.Error(e)
		return e
	}

	if d.categoryTokenizer != nil {
		e = d.categoryTokenizer.Save(filepath.Join(d.cacheDir, "category-tokenizer.json"))
		if e != nil {
			d.errorHandler.Error(e)
			return e
		}
	}

	e = ioutil.WriteFile(filepath.Join(d.cacheDir, cacheFileName), cacheBytes, os.ModePerm)
	if e != nil {
		d.errorHandler.Error(e)
		return e
	}

	d.logger.InfoF("data", "Found %d rows. Got class counts: %#v Got class weights: %#v", d.Count, d.ClassCounts, d.ClassWeights)

	return nil
}

func (d *SingleFileDataset) NumCategoricalClasses() int {
	return len(d.ClassCounts)
}

func (d *SingleFileDataset) Len() int {
	return int(d.limit)
}

func (d *SingleFileDataset) fitPreProcessors() error {
	anyNeedingFit := false
	for _, processor := range d.columnProcessors {
		if processor.RequiresFit {
			e := processor.Load()
			if e == nil {
				d.logger.InfoF("data", "Loaded Pre-Processor: %s", processor.Name)
			} else {
				anyNeedingFit = true
			}
		}
	}

	if !anyNeedingFit {
		d.logger.InfoF("data", "Loaded All Pre-Processors")
		return nil
	}

	d.logger.InfoF("data", "Fitting Pre-Processors")

	d.Shuffle(time.Now().UnixNano())
	d.limit = int32(d.Count)
	lastPrint := time.Now().Unix()
	progress, lastProgress := 0, 0

	var loopError error

	swg := sizedwaitgroup.New(64)
	concurrentFileSwg := sizedwaitgroup.New(int(d.concurrentFileLimit))
	endChan := make(chan bool, 100)
	var errs []error
	var rowsFit int32

	for true {

		select {
		case <-endChan:
			concurrentFileSwg.Wait()
			swg.Wait()
			for _, processor := range d.columnProcessors {
				if processor.RequiresFit {
					e := processor.FinishFit()
					if e != nil {
						return e
					}
				}
			}

			e := d.Reset()
			if e != nil {
				return e
			}

			e = d.Unshuffle()
			if e != nil {
				return e
			}

			e = d.Reset()
			if e != nil {
				return e
			}

			fmt.Println()

			d.logger.InfoF("data", "Fit tokenizers")
			return nil
		default:
			if len(errs) > 0 {
				for _, e := range errs {
					return e
				}
			}
			concurrentFileSwg.Add()
			go func() {
				defer concurrentFileSwg.Done()
				row, e := d.getRow()
				if errors.Is(e, ErrGeneratorEnd) {
					endChan <- true
					return
				}
				for _, processor := range d.columnProcessors {
					if processor.RequiresFit {
						swg.Add()

						go func(processor *preprocessor.Processor, row []string) {
							defer swg.Done()
							if processor.DataLength > 1 {
								if len(row) <= processor.LineOffset+processor.DataLength {
									if d.ignoreParseErrors {
										return
									}
									e = fmt.Errorf("row did not contain enough columns for processor %s at offset %d", processor.Name, processor.LineOffset)
									return
								}
								// TODO: find a nicer way to capture multiple columns in a row
								e = processor.FitString([]string{strings.Join(row[processor.LineOffset:processor.LineOffset+processor.DataLength], ",")})
								if e != nil {
									if d.ignoreParseErrors {
										return
									}
									d.errorHandler.Error(e)
									loopError = e
								}
							} else {
								if len(row) <= processor.LineOffset {
									if d.ignoreParseErrors {
										return
									}
									e = fmt.Errorf("row did not contain enough columns for processor %s at offset %d", processor.Name, processor.LineOffset)
									return
								}
								e = processor.FitString([]string{row[processor.LineOffset]})
								if e != nil {
									if d.ignoreParseErrors {
										return
									}
									d.errorHandler.Error(e)
									loopError = e
								}
							}

						}(processor, row)

						if loopError != nil {
							errs = append(errs, e)
						}
					}
				}

				atomic.AddInt32(&rowsFit, 1)
				fitCount := atomic.LoadInt32(&rowsFit)
				if fitCount > int32(d.maxRowsForFit) {
					endChan <- true
				}

			}()

			now := time.Now().Unix()
			if now > lastPrint {
				lastPrint = now
				fmt.Print(fmt.Sprintf("\rFitting preprocessors: %d %d/s", progress, progress-lastProgress))
				lastProgress = progress
			}
			progress++
		}
	}

	return nil
}

func (d *SingleFileDataset) SetMode(mode GeneratorMode) Dataset {
	d.mode = mode

	if mode == GeneratorModeTrain {
		d.offset = 0
		d.limit = d.trainCount
	} else if mode == GeneratorModeVal {
		d.offset = d.trainCount
		d.limit = d.valCount
	} else if mode == GeneratorModeTest {
		d.offset = d.trainCount + d.valCount
		d.limit = d.testCount
	}
	offset := d.offset
	d.generatorOffset = &offset

	return d
}

func (d *SingleFileDataset) getRow() ([]string, error) {

	if d.shuffled {
		d.generatorOffsetLock.Lock()
		generatorOffset := atomic.LoadInt32(d.generatorOffset)
		atomic.AddInt32(d.generatorOffset, 1)
		d.generatorOffsetLock.Unlock()
		if len(d.lineOffsets) <= int(generatorOffset) {
			return nil, ErrGeneratorEnd
		}
		offset := d.lineOffsets[int(generatorOffset)]
		var file *os.File
		for true {
			file = d.filePool.Get().(*os.File)
			if file == nil {
				cbutil.Sleep(time.Millisecond)
			} else {
				break
			}
		}
		_, e := file.Seek(offset, io.SeekStart)
		if e != nil {
			d.errorHandler.Error(e)
			return nil, e
		}

		reader := csv.NewReader(file)
		line, e := reader.Read()
		d.filePool.Put(file)

		if errors.Is(e, io.EOF) || e == nil {
			return line, nil
		} else {
			return nil, e
		}
	} else {
		panic("Non shuffled mode not implemented")
	}
}

func (d *SingleFileDataset) Shuffle(seed int64) {
	rand.Seed(seed)
	rand.Shuffle(len(d.lineOffsets), func(i, j int) { d.lineOffsets[i], d.lineOffsets[j] = d.lineOffsets[j], d.lineOffsets[i] })
	d.shuffled = true
}

func (d *SingleFileDataset) Unshuffle() error {
	e := d.readLineOffsets()
	if e != nil {
		return e
	}
	d.shuffled = false

	return d.Reset()
}

func (d *SingleFileDataset) GetColumnNames() []string {
	var columnNames []string

	for _, processor := range d.columnProcessors {
		columnNames = append(columnNames, processor.Name)
	}

	return columnNames
}

func (d *SingleFileDataset) GeneratorChan(batchSize int, preFetch int) chan Batch {
	generatorChan := make(chan Batch, preFetch)

	go func() {
		swg := sizedwaitgroup.New(int(d.concurrentFileLimit))
		for i := 0; i < int(d.limit)/batchSize; i++ {
			swg.Add()
			go func() {
				defer swg.Done()
				x, y, classWeights, e := d.Generate(batchSize)

				if errors.Is(e, ErrGeneratorEnd) {
					return
				}
				if e != nil && !errors.Is(e, ErrGeneratorEnd) {
					d.errorHandler.Error(e)
					return
				}

				generatorChan <- Batch{
					X:            x,
					Y:            y,
					ClassWeights: classWeights,
				}
			}()
		}
		swg.Wait()
		close(generatorChan)
	}()

	return generatorChan
}

func (d *SingleFileDataset) Generate(batchSize int) ([]*tf.Tensor, *tf.Tensor, *tf.Tensor, error) {
	var x []*tf.Tensor

	xStrings := make([][]string, len(d.columnProcessors))
	var yInts [][]int32

	for true {
		row, e := d.getRow()
		if errors.Is(e, ErrGeneratorEnd) {
			return nil, nil, nil, e
		}

		if len(row) == 0 {
			continue
		}

		var lineError error
		for offset, processor := range d.columnProcessors {
			if processor.DataLength > 1 {
				if len(row) <= processor.LineOffset+processor.DataLength {
					lineError = fmt.Errorf(
						"row did not contain enough columns for processor %s at offset %d and length %d",
						processor.Name,
						processor.LineOffset,
						processor.DataLength,
					)
				} else {
					// TODO: find a nicer way to capture multiple columns in a row
					xStrings[offset] = append(xStrings[offset], strings.Join(row[processor.LineOffset:processor.LineOffset+processor.DataLength], ","))
				}
			} else {
				if len(row) <= processor.LineOffset {
					lineError = fmt.Errorf(
						"row did not contain enough columns for processor %s at offset %d",
						processor.Name,
						processor.LineOffset,
					)
				} else {
					xStrings[offset] = append(xStrings[offset], row[processor.LineOffset])
				}
			}
		}
		if lineError != nil {
			if d.ignoreParseErrors {
				continue
			}
			d.errorHandler.Error(e)
			return nil, nil, nil, e
		}

		if len(row) <= d.categoryOffset {
			if d.ignoreParseErrors {
				continue
			}
			e = fmt.Errorf("row did not contain enough columns for categoryOffset at %d", d.categoryOffset)
			d.errorHandler.Error(e)
			return nil, nil, nil, e
		}

		var yInt int

		if d.categoryTokenizer != nil {
			yInt = int(d.categoryTokenizer.Tokenize(row[d.categoryOffset])[0])
		} else {
			yInt, e = strconv.Atoi(row[d.categoryOffset])
			if e != nil {
				if d.ignoreParseErrors {
					continue
				}
				d.errorHandler.Error(e)
				return nil, nil, nil, e
			}
		}

		yInts = append(yInts, []int32{int32(yInt)})

		if len(yInts) >= batchSize {
			break
		}
	}

	for offset, processor := range d.columnProcessors {
		process, e := processor.ProcessString(xStrings[offset])
		if e != nil {
			return nil, nil, nil, e
		}

		x = append(x, process)
	}

	var classWeights []float32
	for _, yInt32 := range yInts {
		classWeights = append(classWeights, d.ClassWeights[int(yInt32[0])])
	}

	classWeightsTensor, e := tf.NewTensor(classWeights)
	if e != nil {
		d.errorHandler.Error(e)
		return nil, nil, nil, e
	}

	y, e := tf.NewTensor(yInts)
	if e != nil {
		d.errorHandler.Error(e)
		return nil, nil, nil, e
	}

	return x, y, classWeightsTensor, nil
}

func (d *SingleFileDataset) Reset() error {
	if d.shuffled {
		offset := d.offset
		d.generatorOffset = &offset
	}

	return nil
}

func (d *SingleFileDataset) SaveProcessors(saveDir string) error {
	e := os.MkdirAll(saveDir, os.ModePerm)
	if e != nil {
		d.errorHandler.Error(e)
		return e
	}
	for _, processor := range d.columnProcessors {
		e := processor.Save(saveDir)
		if e != nil {
			return e
		}
	}
	return nil
}

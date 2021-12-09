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
	"time"
)

type SingleFileDataset struct {
	ClassCounts map[int]int
	Count       int

	filePath          string
	file              *os.File
	reader            *csv.Reader
	shuffled          bool
	generatorOffset   int
	cacheDir          string
	categoryOffset    int
	columnProcessors  []*preprocessor.Processor
	lineOffsets       []int64
	trainPercent      float32
	valPercent        float32
	testPercent       float32
	trainCount        int
	valCount          int
	testCount         int
	mode              GeneratorMode
	offset            int
	limit             int
	categoryTokenizer *preprocessor.Tokenizer

	logger       *cblog.Logger
	errorHandler *cberrors.ErrorsContainer
}

func NewSingleFileDataset(
	logger *cblog.Logger,
	errorHandler *cberrors.ErrorsContainer,
	filePath string,
	cacheDir string,
	categoryOffset int,
	trainPercent float32,
	valPercent float32,
	testPercent float32,
	columnProcessors ...*preprocessor.Processor,
) (*SingleFileDataset, error) {

	logger.InfoF("data", "Initialising single file dataset at: %s", filePath)

	file, e := os.Open(filePath)
	if e != nil {
		errorHandler.Error(e)
		return nil, e
	}

	d := &SingleFileDataset{
		logger:           logger,
		errorHandler:     errorHandler,
		filePath:         filePath,
		file:             file,
		reader:           csv.NewReader(file),
		cacheDir:         cacheDir,
		categoryOffset:   categoryOffset,
		columnProcessors: columnProcessors,
		trainPercent:     trainPercent,
		valPercent:       valPercent,
		testPercent:      testPercent,
		ClassCounts:      make(map[int]int),
	}

	e = os.MkdirAll(cacheDir, os.ModePerm)
	if e != nil && e != os.ErrExist {
		errorHandler.Error(e)
		return nil, e
	}

	if _, e := os.Stat(filepath.Join(cacheDir, "category-tokenizer.json")); e == nil {
		d.categoryTokenizer = preprocessor.NewTokenizer(errorHandler, 1, -1, true)
		e = d.categoryTokenizer.Load(filepath.Join(cacheDir, "category-tokenizer.json"))
		if e != nil {
			errorHandler.Error(e)
			return nil, e
		}
	}

	e = d.readLineOffsets()
	if e != nil {
		return nil, e
	}

	d.trainCount = int(math.Ceil(float64(len(d.lineOffsets)) * float64(d.trainPercent)))
	d.valCount = int(math.Ceil(float64(len(d.lineOffsets)) * float64(d.valPercent)))
	d.testCount = int(math.Ceil(float64(len(d.lineOffsets)) * float64(d.testPercent)))

	e = d.fitPreProcessors()
	if e != nil {
		return nil, e
	}

	return d, nil
}

type fileStatsCache struct {
	LineOffsets []int64
	Count       int
	ClassCounts map[int]int
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

		d.logger.InfoF("data", "Found %d rows. Got class counts: %#v", d.Count, d.ClassCounts)

		return nil
	}

	d.logger.InfoF("data", "Reading line offsets and counting stats")

	reader := bufio.NewReader(d.file)

	offset := int64(0)

	lastPrint := time.Now().Unix()
	progress, lastProgress := 0, 0

	for true {
		readBytes, e := reader.ReadBytes('\n')

		if errors.Is(e, io.EOF) {
			break
		}

		offset += int64(len(readBytes))
		d.lineOffsets = append(d.lineOffsets, offset)
		d.Count++

		csvReader := csv.NewReader(bytes.NewBuffer(readBytes))
		line, e := csvReader.Read()
		if e != nil && !errors.Is(e, io.EOF) {
			d.errorHandler.Error(e)
			return e
		}
		if len(line) == 0 {
			continue
		}
		if len(line) < d.categoryOffset {
			e = fmt.Errorf("len(line) (%d) < d.categoryOffset (%d)", len(line), d.categoryOffset)
			d.errorHandler.Error(e)
			return e
		}

		category := line[d.categoryOffset]

		categoryInt, e := strconv.Atoi(category)
		// TODO: this magical behaviour could be nicer
		if e != nil {
			if d.categoryTokenizer == nil {
				d.categoryTokenizer = preprocessor.NewTokenizer(d.errorHandler, 1, -1, true)
			}
			d.categoryTokenizer.Fit(category)
			d.categoryTokenizer.FinishFit()
			categoryInt = int(d.categoryTokenizer.Tokenize(category)[0])
		}

		count := d.ClassCounts[categoryInt]
		count++
		d.ClassCounts[categoryInt] = count

		now := time.Now().Unix()
		if now > lastPrint {
			lastPrint = now
			fmt.Print(fmt.Sprintf("\rReading offsets and counting: %d %d/s", progress, progress-lastProgress))
			lastProgress = progress
		}
		progress++
	}
	fmt.Println()

	cacheBytes, e := json.Marshal(fileStatsCache{
		LineOffsets: d.lineOffsets,
		Count:       d.Count,
		ClassCounts: d.ClassCounts,
	})
	if e != nil {
		d.errorHandler.Error(e)
		return e
	}

	if d.categoryTokenizer != nil {
		d.categoryTokenizer.FinishFit()
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

	d.logger.InfoF("data", "Found %d rows. Got class counts: %#v", d.Count, d.ClassCounts)

	return nil
}

func (d *SingleFileDataset) Len() int {
	return d.limit
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
	d.limit = d.Count
	lastPrint := time.Now().Unix()
	progress, lastProgress := 0, 0

	var loopError error

	swg := sizedwaitgroup.New(64)

	for i := 0; i < 1000000; i++ {
		row, e := d.getRow()
		if errors.Is(e, ErrGeneratorEnd) {
			break
		}
		for _, processor := range d.columnProcessors {
			if processor.RequiresFit {
				swg.Add()

				go func(processor *preprocessor.Processor) {
					defer swg.Done()
					if processor.DataLength > 1 {
						if len(row) <= processor.LineOffset+processor.DataLength {
							e = fmt.Errorf("row did not contain enough columns for processor %s at offset %d", processor.Name, processor.LineOffset)
							return
						}
						// TODO: find a nicer way to capture multiple columns in a row
						e = processor.Fit([]string{strings.Join(row[processor.LineOffset:processor.LineOffset+processor.DataLength], ",")})
						if e != nil {
							d.errorHandler.Error(e)
							loopError = e
						}
					} else {
						if len(row) <= processor.LineOffset {
							e = fmt.Errorf("row did not contain enough columns for processor %s at offset %d", processor.Name, processor.LineOffset)
							return
						}
						e = processor.Fit([]string{row[processor.LineOffset]})
						if e != nil {
							d.errorHandler.Error(e)
							loopError = e
						}
					}

				}(processor)

				if loopError != nil {
					return e
				}
			}
		}

		now := time.Now().Unix()
		if now > lastPrint {
			lastPrint = now
			fmt.Print(fmt.Sprintf("\rFitting preprocessors: %d %d/s", progress, progress-lastProgress))
			lastProgress = progress
		}
		progress++
	}
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
	d.generatorOffset = d.offset

	return d
}

func (d *SingleFileDataset) getRow() ([]string, error) {

	if d.shuffled {
		if len(d.lineOffsets) <= d.generatorOffset {
			return nil, ErrGeneratorEnd
		}
		offset := d.lineOffsets[d.generatorOffset]
		_, e := d.file.Seek(offset, io.SeekStart)
		if e != nil {
			d.errorHandler.Error(e)
			return nil, e
		}

		d.reader = csv.NewReader(d.file)
		line, e := d.reader.Read()

		d.generatorOffset++

		if d.generatorOffset >= d.offset+d.limit {
			return line, ErrGeneratorEnd
		}

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

	// TODO: change class weights to a single tensor
	var posWeight, negWeight float32
	if d.ClassCounts[1] > d.ClassCounts[0] {
		posWeight = float32(d.ClassCounts[0]) / float32(d.ClassCounts[1])
		negWeight = 1
	} else {
		posWeight = 1
		negWeight = float32(d.ClassCounts[1]) / float32(d.ClassCounts[0])
	}

	posWeightTensor, e := tf.NewTensor(posWeight)
	if e != nil {
		d.errorHandler.Error(e)
		return nil
	}

	negWeightTensor, e := tf.NewTensor(negWeight)
	if e != nil {
		d.errorHandler.Error(e)
		return nil
	}

	go func() {
		for true {
			x, y, e := d.Generate(batchSize)

			if errors.Is(e, ErrGeneratorEnd) {
				close(generatorChan)
				break
			}
			if e != nil && !errors.Is(e, ErrGeneratorEnd) {
				d.errorHandler.Error(e)
				close(generatorChan)
				break
			}

			generatorChan <- Batch{
				X:         x,
				Y:         y,
				PosWeight: posWeightTensor,
				NegWeight: negWeightTensor,
			}
		}
	}()

	return generatorChan
}

func (d *SingleFileDataset) Generate(batchSize int) ([]*tf.Tensor, *tf.Tensor, error) {
	var x []*tf.Tensor

	xStrings := make([][]string, len(d.columnProcessors))
	var yInts [][]int32

	for true {
		row, e := d.getRow()
		if errors.Is(e, ErrGeneratorEnd) {
			return nil, nil, e
		}

		if len(row) == 0 {
			continue
		}

		var lineError bool
		for offset, processor := range d.columnProcessors {
			if processor.DataLength > 1 {
				if len(row) <= processor.LineOffset+processor.DataLength {
					lineError = true
				} else {
					// TODO: find a nicer way to capture multiple columns in a row
					xStrings[offset] = append(xStrings[offset], strings.Join(row[processor.LineOffset:processor.LineOffset+processor.DataLength], ","))
				}
			} else {
				if len(row) <= processor.LineOffset {
					lineError = true
				} else {
					xStrings[offset] = append(xStrings[offset], row[processor.LineOffset])
				}
			}
		}
		if lineError {
			continue
		}

		if len(row) <= d.categoryOffset {
			e = fmt.Errorf("row did not contain enough columns for categoryOffset at %d", d.categoryOffset)
			d.errorHandler.Error(e)
			return nil, nil, e
		}

		var yInt int

		if d.categoryTokenizer != nil {
			yInt = int(d.categoryTokenizer.Tokenize(row[d.categoryOffset])[0])
		} else {
			yInt, e = strconv.Atoi(row[d.categoryOffset])
			if e != nil {
				d.errorHandler.Error(e)
				return nil, nil, e
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
			return nil, nil, e
		}

		x = append(x, process)
	}

	y, e := tf.NewTensor(yInts)
	if e != nil {
		d.errorHandler.Error(e)
		return nil, nil, e
	}

	return x, y, nil
}

func (d *SingleFileDataset) Reset() error {
	if d.shuffled {
		d.generatorOffset = d.offset
	} else {
		_, e := d.file.Seek(0, io.SeekStart)
		if e != nil {
			d.errorHandler.Error(e)
			return e
		}

		d.reader = csv.NewReader(d.file)
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

package data

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/codingbeard/cberrors"
	"github.com/codingbeard/cblog"
	"github.com/codingbeard/tfkg/preprocessor"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"github.com/remeh/sizedwaitgroup"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

type imgMetadata struct {
	filepath string
	category int
}

type ImgFolderDataset struct {
	ClassCounts  map[int]int
	ClassWeights map[int]float32
	Count        int

	classCountsLock     *sync.Mutex
	parentDir           string
	filePool            *sync.Pool
	cacheDir            string
	concurrentFileLimit int32
	openFileCount       *int32
	ignoreParseErrors   bool
	shuffled            bool
	generatorOffset     int
	processor           *preprocessor.Processor
	images              []imgMetadata
	trainPercent        float32
	valPercent          float32
	testPercent         float32
	trainCount          int
	valCount            int
	testCount           int
	mode                GeneratorMode
	offset              int
	limit               int
	categoryTokenizer   *preprocessor.Tokenizer

	logger       *cblog.Logger
	errorHandler *cberrors.ErrorsContainer
}

type ImgFolderDatasetConfig struct {
	ParentDir           string
	CacheDir            string
	TrainPercent        float32
	ValPercent          float32
	TestPercent         float32
	IgnoreParseErrors   bool
	ConcurrentFileLimit int32
	ClassWeights        map[int]float32
}

func NewImgFolderDataset(
	logger *cblog.Logger,
	errorHandler *cberrors.ErrorsContainer,
	config ImgFolderDatasetConfig,
	processor *preprocessor.Processor,
) (*ImgFolderDataset, error) {

	logger.InfoF("data", "Initialising single file dataset at: %s", config.ParentDir)

	stat, e := os.Stat(config.ParentDir)
	if e != nil {
		errorHandler.Error(e)
		return nil, e
	}
	if !stat.IsDir() {
		e = fmt.Errorf("%s is not a directory", config.ParentDir)
	}
	if config.ConcurrentFileLimit == 0 {
		config.ConcurrentFileLimit = 1
	}

	if config.ClassWeights == nil {
		config.ClassWeights = make(map[int]float32)
	}

	var openFileCount int32

	d := &ImgFolderDataset{
		logger:              logger,
		errorHandler:        errorHandler,
		classCountsLock:     &sync.Mutex{},
		parentDir:           config.ParentDir,
		cacheDir:            config.CacheDir,
		ignoreParseErrors:   config.IgnoreParseErrors,
		processor:           processor,
		trainPercent:        config.TrainPercent,
		valPercent:          config.ValPercent,
		testPercent:         config.TestPercent,
		ClassCounts:         make(map[int]int),
		ClassWeights:        config.ClassWeights,
		concurrentFileLimit: config.ConcurrentFileLimit,
		openFileCount:       &openFileCount,
		generatorOffset:     0,
	}

	d.filePool = &sync.Pool{
		New: func() interface{} {
			currentFileCount := atomic.LoadInt32(d.openFileCount)
			if currentFileCount >= d.concurrentFileLimit {
				return nil
			}

			file, e := os.Open(d.parentDir)
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

	e = d.readFileNames()
	if e != nil {
		return nil, e
	}

	d.trainCount = int(math.Ceil(float64(len(d.images)) * float64(d.trainPercent)))
	d.valCount = int(math.Ceil(float64(len(d.images)) * float64(d.valPercent)))
	d.testCount = int(math.Ceil(float64(len(d.images)) * float64(d.testPercent)))

	return d, nil
}

type imgStatsCache struct {
	Images       []imgMetadata
	Count        int
	ClassCounts  map[int]int
	ClassWeights map[int]float32
}

func (d *ImgFolderDataset) readFileNames() error {
	cacheFileName := "file-stats.json"
	cacheFileBytes, e := ioutil.ReadFile(filepath.Join(d.cacheDir, cacheFileName))
	if e != nil && !errors.Is(e, os.ErrNotExist) {
		d.errorHandler.Error(e)
		return e
	} else if e == nil {
		var cache imgStatsCache
		e = json.Unmarshal(cacheFileBytes, &cache)
		if e != nil {
			d.errorHandler.Error(e)
			return e
		}
		d.logger.InfoF("data", "Loading image file paths and stats from cache file")

		d.images = cache.Images
		d.Count = cache.Count
		d.ClassCounts = cache.ClassCounts
		if len(d.ClassWeights) == 0 {
			d.ClassWeights = cache.ClassWeights
		}

		d.logger.InfoF("data", "Found %d rows. Got class counts: %#v Got class weights: %#v", d.Count, d.ClassCounts, d.ClassWeights)

		return nil
	}

	d.logger.InfoF("data", "Reading image file paths and counting stats")

	lastPrint := time.Now().Unix()
	progress, lastProgress := 0, 0
	categoryFolders, e := filepath.Glob(filepath.Join(d.parentDir, "*"))
	if e != nil {
		d.errorHandler.Error(e)
		return e
	}
	for _, folder := range categoryFolders {
		category := filepath.Base(folder)
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

		imagePaths, e := filepath.Glob(filepath.Join(folder, "*"))
		if e != nil {
			d.errorHandler.Error(e)
			return e
		}
		for _, imgFilePath := range imagePaths {
			d.classCountsLock.Lock()
			count := d.ClassCounts[categoryInt]
			count++
			d.ClassCounts[categoryInt] = count
			d.classCountsLock.Unlock()

			d.Count++

			d.images = append(d.images, imgMetadata{
				filepath: imgFilePath,
				category: categoryInt,
			})

			now := time.Now().Unix()
			if now > lastPrint {
				lastPrint = now
				fmt.Print(fmt.Sprintf("\rReading image file paths and counting: %d %d/s", progress, progress-lastProgress))
				lastProgress = progress
			}
			progress++
		}
	}
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

	cacheBytes, e := json.Marshal(imgStatsCache{
		Images:       d.images,
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

	d.logger.InfoF("data", "Found %d images. Got class counts: %#v Got class weights: %#v", d.Count, d.ClassCounts, d.ClassWeights)

	return nil
}

func (d *ImgFolderDataset) NumCategoricalClasses() int {
	return len(d.ClassCounts)
}

func (d *ImgFolderDataset) Len() int {
	return int(d.limit)
}

func (d *ImgFolderDataset) SetMode(mode GeneratorMode) Dataset {
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

func (d *ImgFolderDataset) getRow() (string, int, error) {
	if d.shuffled {
		if len(d.images) <= d.generatorOffset {
			return "", 0, ErrGeneratorEnd
		}
		img := d.images[d.generatorOffset]
		d.generatorOffset++
		return img.filepath, img.category, nil
	} else {
		panic("Non shuffled mode not implemented")
	}
}

func (d *ImgFolderDataset) Shuffle(seed int64) {
	rand.Seed(seed)
	rand.Shuffle(len(d.images), func(i, j int) { d.images[i], d.images[j] = d.images[j], d.images[i] })
	d.shuffled = true
}

func (d *ImgFolderDataset) Unshuffle() error {
	e := d.readFileNames()
	if e != nil {
		return e
	}
	d.shuffled = false

	return d.Reset()
}

func (d *ImgFolderDataset) GetColumnNames() []string {
	return []string{d.processor.Name}
}

func (d *ImgFolderDataset) GeneratorChan(batchSize int, preFetch int) chan Batch {
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

func (d *ImgFolderDataset) Generate(batchSize int) ([]*tf.Tensor, *tf.Tensor, *tf.Tensor, error) {
	var x []*tf.Tensor

	var xPaths []string
	var yInts [][]int32

	for true {
		filePath, category, e := d.getRow()
		if errors.Is(e, ErrGeneratorEnd) {
			return nil, nil, nil, e
		}

		if len(filePath) == 0 {
			continue
		}

		xPaths = append(xPaths, filePath)

		yInts = append(yInts, []int32{int32(category)})

		if len(yInts) >= batchSize {
			break
		}
	}

	process, e := d.processor.ProcessString(xPaths)
	if e != nil {
		return nil, nil, nil, e
	}

	x = append(x, process)

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

func (d *ImgFolderDataset) Reset() error {
	if d.shuffled {
		d.generatorOffset = d.offset
	}

	return nil
}

func (d *ImgFolderDataset) SaveProcessors(saveDir string) error {
	return nil
}

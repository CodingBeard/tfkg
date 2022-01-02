package data

import (
	"errors"
	"fmt"
	"github.com/codingbeard/cberrors"
	"github.com/codingbeard/cblog"
	"github.com/codingbeard/tfkg/preprocessor"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"github.com/remeh/sizedwaitgroup"
	"math"
	"math/rand"
	"runtime"
	"time"
)

type ValuesDataset struct {
	ClassCounts  map[int]int
	ClassWeights map[int]float32
	Count        int

	shuffled         bool
	generatorOffset  int
	cacheDir         string
	yProcessor       *preprocessor.Processor
	isCategorical    bool
	columnProcessors []*preprocessor.Processor
	trainPercent     float32
	valPercent       float32
	testPercent      float32
	trainCount       int
	valCount         int
	testCount        int
	mode             GeneratorMode
	offsets          []int
	offset           int
	limit            int
	xValues          [][]interface{}
	yValues          []interface{}

	logger       *cblog.Logger
	errorHandler *cberrors.ErrorsContainer
}

type ValuesDatasetConfig struct {
	CacheDir     string
	TrainPercent float32
	ValPercent   float32
	TestPercent  float32
}

func NewValuesDataset(
	logger *cblog.Logger,
	errorHandler *cberrors.ErrorsContainer,
	config ValuesDatasetConfig,
	yProcessor *preprocessor.Processor,
	columnProcessors ...*preprocessor.Processor,
) (*ValuesDataset, error) {

	logger.InfoF("data", "Initialising values dataset")

	d := &ValuesDataset{
		logger:           logger,
		errorHandler:     errorHandler,
		yProcessor:       yProcessor,
		columnProcessors: columnProcessors,
		trainPercent:     config.TrainPercent,
		valPercent:       config.ValPercent,
		testPercent:      config.TestPercent,
		ClassCounts:      make(map[int]int),
		ClassWeights:     make(map[int]float32),
	}

	return d, nil
}

func (d *ValuesDataset) SetValues(yValues []interface{}, xValues ...[]interface{}) error {
	for offset := range xValues {
		if len(d.columnProcessors) <= offset {
			e := fmt.Errorf(
				"there were not enough column processors defined on ValuesDataset (%d) for the number of input values (%d)",
				len(d.columnProcessors),
				len(xValues),
			)
			d.errorHandler.Error(e)
			return e
		}
	}

	d.Count = len(yValues)

	for i := 0; i < len(yValues); i++ {
		d.offsets = append(d.offsets, i)
	}

	d.isCategorical = false
	for _, value := range yValues {
		intValue, ok := value.(int32)
		if ok {
			d.isCategorical = true
			count := d.ClassCounts[int(intValue)]
			count++
			d.ClassCounts[int(intValue)] = count
		}
	}

	if d.isCategorical {
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

	d.yValues = yValues
	d.xValues = xValues

	d.trainCount = int(math.Ceil(float64(len(d.offsets)) * float64(d.trainPercent)))
	d.valCount = int(math.Ceil(float64(len(d.offsets)) * float64(d.valPercent)))
	d.testCount = int(math.Ceil(float64(len(d.offsets)) * float64(d.testPercent)))

	d.logger.InfoF("data", "Got %d rows. Got class counts: %#v Got class weights: %#v", d.Count, d.ClassCounts, d.ClassWeights)

	return d.fitPreProcessors()
}

func (d *ValuesDataset) NumCategoricalClasses() int {
	return len(d.ClassCounts)
}

func (d *ValuesDataset) Len() int {
	return d.limit
}

func (d *ValuesDataset) fitPreProcessors() error {
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
		xInterface, _, e := d.getRow()
		if errors.Is(e, ErrGeneratorEnd) {
			break
		}
		for _, processor := range d.columnProcessors {
			if processor.RequiresFit {
				swg.Add()

				go func(processor *preprocessor.Processor) {
					defer swg.Done()
					if len(xInterface) <= processor.LineOffset {
						e = fmt.Errorf("row did not contain enough columns for processor %s at offset %d", processor.Name, processor.LineOffset)
						return
					}
					e = processor.FitInterface(xInterface[processor.LineOffset])
					if e != nil {
						d.errorHandler.Error(e)
						loopError = e
					}
				}(processor)

				if loopError != nil {
					return e
				}
			}
		}
		swg.Wait()

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

func (d *ValuesDataset) SetMode(mode GeneratorMode) Dataset {
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

func (d *ValuesDataset) getRow() ([]interface{}, interface{}, error) {
	if len(d.yValues) <= d.generatorOffset {
		return nil, nil, ErrGeneratorEnd
	}

	offset := d.offsets[d.generatorOffset]

	var x []interface{}
	for i := range d.columnProcessors {
		x = append(x, d.xValues[i][offset])
	}
	y := d.yValues[offset]

	d.generatorOffset++

	if d.generatorOffset >= d.offset+d.limit {
		return x, y, ErrGeneratorEnd
	}
	return x, y, nil
}

func (d *ValuesDataset) Shuffle(seed int64) {
	rand.Seed(seed)
	rand.Shuffle(len(d.offsets), func(i, j int) { d.offsets[i], d.offsets[j] = d.offsets[j], d.offsets[i] })
	d.shuffled = true
}

func (d *ValuesDataset) Unshuffle() error {
	d.offsets = []int{}
	for i := 0; i < len(d.yValues); i++ {
		d.offsets = append(d.offsets, i)
	}
	d.shuffled = false

	return d.Reset()
}

func (d *ValuesDataset) GetColumnNames() []string {
	var columnNames []string

	for _, processor := range d.columnProcessors {
		columnNames = append(columnNames, processor.Name)
	}

	return columnNames
}

func (d *ValuesDataset) GeneratorChan(batchSize int, preFetch int) chan Batch {
	generatorChan := make(chan Batch, preFetch)

	go func() {
		swg := sizedwaitgroup.New(runtime.NumCPU())
		for i := 0; i < d.limit/batchSize; i++ {
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

func (d *ValuesDataset) Generate(batchSize int) ([]*tf.Tensor, *tf.Tensor, *tf.Tensor, error) {
	var x []*tf.Tensor

	xRaw := make([][]interface{}, len(d.columnProcessors))
	var yRaw []interface{}

	for true {
		xInterfaces, yInterface, e := d.getRow()
		if errors.Is(e, ErrGeneratorEnd) {
			return nil, nil, nil, e
		}

		if len(xInterfaces) == 0 {
			continue
		}

		for i := range d.columnProcessors {
			xRaw[i] = append(xRaw[i], xInterfaces[i])
		}

		yRaw = append(yRaw, yInterface)

		if len(yRaw) >= batchSize {
			break
		}
	}

	for offset, processor := range d.columnProcessors {
		process, e := processor.ProcessInterface(xRaw[offset])
		if e != nil {
			return nil, nil, nil, e
		}

		x = append(x, process)
	}

	y, e := d.yProcessor.ProcessInterface(yRaw)
	if e != nil {
		d.errorHandler.Error(e)
		return nil, nil, nil, e
	}

	var classWeights []float32
	categoricalY, ok := y.Value().([][]int32)
	if ok {
		for _, yInt32 := range categoricalY {
			classWeights = append(classWeights, d.ClassWeights[int(yInt32[0])])
		}
	} else {
		for range yRaw {
			classWeights = append(classWeights, 1)
		}
	}

	classWeightsTensor, e := tf.NewTensor(classWeights)
	if e != nil {
		d.errorHandler.Error(e)
		return nil, nil, nil, e
	}

	return x, y, classWeightsTensor, nil
}

func (d *ValuesDataset) Reset() error {
	d.generatorOffset = d.offset

	return nil
}

func (d *ValuesDataset) SaveProcessors(saveDir string) error {

	return nil
}

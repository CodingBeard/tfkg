package preprocessor

import (
	"fmt"
	"github.com/codingbeard/cberrors"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"image"
	"os"
	"path/filepath"
)

type Processor struct {
	Name       string
	LineOffset int
	// TODO: find a nicer way to capture multiple columns in a row
	DataLength  int
	RequiresFit bool

	cacheDir  string
	divisor   *RegressionDivisor
	tokenizer *Tokenizer
	image     *Image
	reader    func(column []string) interface{}
	converter func(column interface{}) (*tf.Tensor, error)

	errorHandler *cberrors.ErrorsContainer
}

type ProcessorConfig struct {
	CacheDir    string
	LineOffset  int
	DataLength  int
	RequiresFit bool
	Divisor     *RegressionDivisor
	Tokenizer   *Tokenizer
	Image       *Image
	Reader      func(column []string) interface{}
	Converter   func(column interface{}) (*tf.Tensor, error)
}

func NewProcessor(
	errorHandler *cberrors.ErrorsContainer,
	name string,
	config ProcessorConfig,
) *Processor {
	return &Processor{
		errorHandler: errorHandler,
		Name:         name,
		cacheDir:     config.CacheDir,
		LineOffset:   config.LineOffset,
		DataLength:   config.DataLength,
		RequiresFit:  config.RequiresFit,
		divisor:      config.Divisor,
		tokenizer:    config.Tokenizer,
		image:        config.Image,
		reader:       config.Reader,
		converter:    config.Converter,
	}
}

func NewSparseCategoricalTokenizingYProcessor(
	errorHandler *cberrors.ErrorsContainer,
	cacheDir string,
	lineOffset int,
) *Processor {
	return &Processor{
		errorHandler: errorHandler,
		Name:         "y",
		cacheDir:     cacheDir,
		LineOffset:   lineOffset,
		RequiresFit:  true,
		tokenizer: NewTokenizer(errorHandler, 1, -1, TokenizerConfig{
			IsCategoryTokenizer: true,
			DisableFiltering:    true,
		}),
		reader:    ReadStringNop,
		converter: ConvertTokenizerToInt32SliceTensor,
	}
}

func NewSparseCategoricalYProcessor(
	errorHandler *cberrors.ErrorsContainer,
	cacheDir string,
	lineOffset int,
) *Processor {
	return &Processor{
		errorHandler: errorHandler,
		Name:         "y",
		cacheDir:     cacheDir,
		LineOffset:   lineOffset,
		RequiresFit:  false,
		reader:       ReadCsvInt32s,
		converter:    ConvertInt32SliceToTensor,
	}
}

func NewBinaryTokenizingYProcessor(
	errorHandler *cberrors.ErrorsContainer,
	cacheDir string,
	lineOffset int,
) *Processor {
	return &Processor{
		errorHandler: errorHandler,
		Name:         "y",
		cacheDir:     cacheDir,
		LineOffset:   lineOffset,
		RequiresFit:  true,
		tokenizer: NewTokenizer(errorHandler, 1, -1, TokenizerConfig{
			IsCategoryTokenizer: true,
			DisableFiltering:    true,
		}),
		reader:    ReadStringNop,
		converter: ConvertTokenizerToInt32SliceTensor,
	}
}

func NewBinaryYProcessor(
	errorHandler *cberrors.ErrorsContainer,
	cacheDir string,
	lineOffset int,
) *Processor {
	return &Processor{
		errorHandler: errorHandler,
		Name:         "y",
		cacheDir:     cacheDir,
		LineOffset:   lineOffset,
		RequiresFit:  false,
		reader:       ReadCsvInt32s,
		converter:    ConvertInt32SliceToTensor,
	}
}

func (p *Processor) Tokenizer() *Tokenizer {
	return p.tokenizer
}

func (p *Processor) FitString(column []string) error {
	value := p.reader(column)
	if p.divisor != nil {
		floatValues, ok := value.([][]float32)
		if !ok {
			e := fmt.Errorf("error casting read value to []float32 for preprocessor %s, value was: %#v", p.Name, value)
			p.errorHandler.Error(e)
			return e
		}
		for _, floatValue := range floatValues {
			p.divisor.Fit(floatValue)
		}
	} else if p.tokenizer != nil {
		stringValues, ok := value.([]string)
		if !ok {
			e := fmt.Errorf("error casting read value to string for preprocessor %s, value was: %#v", p.Name, value)
			p.errorHandler.Error(e)
			return e
		}
		for _, stringValue := range stringValues {
			p.tokenizer.Fit(stringValue)
		}
	}
	return nil
}

func (p *Processor) FitInterface(column interface{}) error {
	if p.divisor != nil {
		floatValues, ok := column.([][]float32)
		if !ok {
			e := fmt.Errorf("error casting read value to []float32 for preprocessor %s, value was: %#v", p.Name, column)
			p.errorHandler.Error(e)
			return e
		}
		for _, floatValue := range floatValues {
			p.divisor.Fit(floatValue)
		}
	} else if p.tokenizer != nil {
		stringValues, ok := column.([]string)
		if !ok {
			e := fmt.Errorf("error casting read value to string for preprocessor %s, value was: %#v", p.Name, column)
			p.errorHandler.Error(e)
			return e
		}
		for _, stringValue := range stringValues {
			p.tokenizer.Fit(stringValue)
		}
	}
	return nil
}

func (p *Processor) SetLoadDir(dir string) {
	p.cacheDir = dir
}

func (p *Processor) Load() error {
	divisorConfigPath := filepath.Join(p.cacheDir, fmt.Sprintf("%s-divisor.json", p.Name))
	tokenizerConfigPath := filepath.Join(p.cacheDir, fmt.Sprintf("%s-tokenizer.json", p.Name))
	_, e := os.Stat(divisorConfigPath)
	if e == nil {
		p.divisor = NewDivisor(p.errorHandler)
	} else {
		_, e := os.Stat(tokenizerConfigPath)
		if e == nil {
			p.tokenizer = NewTokenizer(p.errorHandler, -1, -1)
		}
	}
	if p.divisor != nil {
		e := p.divisor.Load(divisorConfigPath)
		if e != nil {
			return e
		}
	} else if p.tokenizer != nil {
		e := p.tokenizer.Load(tokenizerConfigPath)
		if e != nil {
			return e
		}
	}
	return nil
}

func (p *Processor) FinishFit() error {
	if p.tokenizer != nil {
		p.tokenizer.FinishFit()
	}
	return p.Save(p.cacheDir)
}

func (p *Processor) Save(saveDir string) error {
	if p.divisor != nil {
		e := p.divisor.Save(filepath.Join(saveDir, fmt.Sprintf("%s-divisor.json", p.Name)))
		if e != nil {
			p.errorHandler.Error(e)
			return e
		}
	} else if p.tokenizer != nil {
		e := p.tokenizer.Save(filepath.Join(saveDir, fmt.Sprintf("%s-tokenizer.json", p.Name)))
		if e != nil {
			p.errorHandler.Error(e)
			return e
		}
	}

	return nil
}

func (p *Processor) ProcessString(columnRows []string) (*tf.Tensor, error) {
	read := p.reader(columnRows)
	if p.divisor != nil {
		var dividedRows [][]float32
		for _, columnRow := range read.([][]float32) {
			divided, e := p.divisor.Divide(columnRow)
			if e != nil {
				p.errorHandler.Error(e)
				return nil, e
			}
			dividedRows = append(dividedRows, divided)
		}
		return p.converter(dividedRows)
	} else if p.tokenizer != nil {
		var tokenizedStrings [][]int32
		for _, columnRow := range read.([]string) {
			tokenized := p.tokenizer.Tokenize(columnRow)
			tokenizedStrings = append(tokenizedStrings, tokenized)
		}
		return p.converter(tokenizedStrings)
	} else if p.image != nil {
		var processedImages []ProcessedImage
		for _, img := range read.([]image.Image) {
			processedImage, e := p.image.Process(img)
			if e != nil {
				p.errorHandler.Error(e)
				return nil, e
			}
			processedImages = append(processedImages, processedImage)
		}
		return p.converter(processedImages)
	}

	return p.converter(read)
}

func (p *Processor) ProcessInterface(columnRows interface{}) (*tf.Tensor, error) {
	if p.reader != nil {
		return p.ProcessString(columnRows.([]string))
	}
	if p.divisor != nil {
		var dividedRows [][]float32
		literalType, ok := columnRows.([][]float32)
		if ok {
			for _, columnRow := range literalType {
				divided, e := p.divisor.Divide(columnRow)
				if e != nil {
					p.errorHandler.Error(e)
					return nil, e
				}
				dividedRows = append(dividedRows, divided)
			}
		} else {
			for _, columnRow := range columnRows.([]interface{}) {
				divided, e := p.divisor.Divide(columnRow.([]float32))
				if e != nil {
					p.errorHandler.Error(e)
					return nil, e
				}
				dividedRows = append(dividedRows, divided)
			}
		}
		return p.converter(dividedRows)
	} else if p.tokenizer != nil {
		var tokenizedStrings [][]int32
		literalType, ok := columnRows.([]string)
		if ok {
			for _, columnRow := range literalType {
				tokenized := p.tokenizer.Tokenize(columnRow)
				tokenizedStrings = append(tokenizedStrings, tokenized)
			}
		} else {
			for _, columnRow := range columnRows.([]interface{}) {
				tokenized := p.tokenizer.Tokenize(columnRow.(string))
				tokenizedStrings = append(tokenizedStrings, tokenized)
			}
		}
		return p.converter(tokenizedStrings)
	} else if p.image != nil {
		var processedImages []ProcessedImage
		literalType, ok := columnRows.([]image.Image)
		if ok {
			for _, img := range literalType {
				processedImage, e := p.image.Process(img)
				if e != nil {
					p.errorHandler.Error(e)
					return nil, e
				}
				processedImages = append(processedImages, processedImage)
			}
		} else {
			for _, img := range columnRows.([]interface{}) {
				processedImage, e := p.image.Process(img.(image.Image))
				if e != nil {
					p.errorHandler.Error(e)
					return nil, e
				}
				processedImages = append(processedImages, processedImage)
			}
		}
		return p.converter(processedImages)
	}

	return p.converter(columnRows)
}

package preprocessor

import (
	"fmt"
	"github.com/codingbeard/cberrors"
	tf "github.com/galeone/tensorflow/tensorflow/go"
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
		reader:       config.Reader,
		converter:    config.Converter,
	}
}

func (p *Processor) Fit(column []string) error {
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

func (p *Processor) SetLoadDir(dir string) {
	p.cacheDir = dir
}

func (p *Processor) Load() error {
	if p.divisor != nil {
		e := p.divisor.Load(filepath.Join(p.cacheDir, fmt.Sprintf("%s-divisor.json", p.Name)))
		if e != nil {
			return e
		}
	} else if p.tokenizer != nil {
		e := p.tokenizer.Load(filepath.Join(p.cacheDir, fmt.Sprintf("%s-tokenizer.json", p.Name)))
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
	}

	return p.converter(read)
}

func (p *Processor) ProcessInterface(columnRows interface{}) (*tf.Tensor, error) {
	if p.reader != nil {
		return p.ProcessString(columnRows.([]string))
	}
	if p.divisor != nil {
		var dividedRows [][]float32
		for _, columnRow := range columnRows.([][]float32) {
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
		for _, columnRow := range columnRows.([]string) {
			tokenized := p.tokenizer.Tokenize(columnRow)
			tokenizedStrings = append(tokenizedStrings, tokenized)
		}
		return p.converter(tokenizedStrings)
	}

	return p.converter(columnRows)
}

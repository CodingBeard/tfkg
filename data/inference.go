package data

import (
	"fmt"
	"github.com/codingbeard/cberrors"
	"github.com/codingbeard/cblog"
	"github.com/codingbeard/tfkg/preprocessor"
	tf "github.com/galeone/tensorflow/tensorflow/go"
)

type Inference struct {
	processorsSaveDir string
	columnProcessors  []*preprocessor.Processor
	categoryTokenizer *preprocessor.Tokenizer

	logger       *cblog.Logger
	errorHandler *cberrors.ErrorsContainer
}

func NewInference(
	logger *cblog.Logger,
	errorHandler *cberrors.ErrorsContainer,
	processorsSaveDir string,
	columnProcessors ...*preprocessor.Processor,
) (*Inference, error) {
	logger.InfoF("data", "Initialising inference provider with processors loaded from: %s", processorsSaveDir)

	for _, processor := range columnProcessors {
		processor.SetLoadDir(processorsSaveDir)
		e := processor.Load()
		if e != nil {
			errorHandler.Error(e)
			return nil, e
		}
	}

	d := &Inference{
		logger:            logger,
		errorHandler:      errorHandler,
		processorsSaveDir: processorsSaveDir,
		columnProcessors:  columnProcessors,
	}

	return d, nil
}

func (d *Inference) GetColumnNames() []string {
	var columnNames []string

	for _, processor := range d.columnProcessors {
		columnNames = append(columnNames, processor.Name)
	}

	return columnNames
}

func (d *Inference) GenerateInputs(rawInputValues ...interface{}) ([]*tf.Tensor, error) {
	var x []*tf.Tensor

	for offset, value := range rawInputValues {
		if len(d.columnProcessors) <= offset {
			e := fmt.Errorf(
				"there were not enough column processors defined on inference (%d) for the number of input values (%d)",
				len(d.columnProcessors),
				len(rawInputValues),
			)
			d.errorHandler.Error(e)
			return nil, e
		}
		processedTensor, e := d.columnProcessors[offset].ProcessInterface(value)
		if e != nil {
			return nil, e
		}

		x = append(x, processedTensor)
	}

	return x, nil
}

package layer

import (
	"fmt"
	tf "github.com/codingbeard/tfkg/tensorflow/go"
	"github.com/codingbeard/tfkg/tensorflow/go/op"
	"github.com/codingbeard/tfkg/util"
)

type Input struct {
	outputOp tf.Output
	shape    tf.Shape

	config InputConfig
	es     []error
}

type InputConfig struct {
	Name  string
	Dtype tf.DataType
	Shape tf.Shape
}

func NewInput(config InputConfig) *Input {
	var es []error
	if config.Name == "" {
		config.Name = util.UniqueName("input")
	}
	if !util.IsValidDtype(config.Dtype) {
		es = append(es, util.NewError(fmt.Errorf("no dtype specified for input %s", config.Name)))
	}
	if config.Shape.NumDimensions() <= 0 {
		es = append(es, util.NewError(fmt.Errorf("no shape specified for input %s", config.Name)))
	}
	return &Input{
		config: config,
		shape:  config.Shape,
		es:     es,
	}
}

func (i *Input) Errors() []error {
	return i.es
}

func (i *Input) Compile(scope *op.Scope) []error {
	if len(i.Errors()) != 0 {
		return i.Errors()
	}

	i.outputOp = op.Placeholder(scope.SubScope(i.config.Name), i.config.Dtype)

	return nil
}

func (i *Input) Output() tf.Output {
	return i.outputOp
}

func (i *Input) Shape() tf.Shape {
	return i.shape
}

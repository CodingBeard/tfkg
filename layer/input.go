package layer

import (
	"fmt"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"strings"
)

type Input struct {
	shape     tf.Shape
	batchSize int
	name      string
	dtype     DataType
	sparse    TfBool
	ragged    TfBool
}

type InputConfig struct {
	BatchSize int
	Name      string
	Sparse    TfBool
	Ragged    TfBool
}

func NewInput(shape tf.Shape, dType DataType, optionalConfig ...InputConfig) *Input {
	var config InputConfig
	if len(optionalConfig) == 1 {
		config = optionalConfig[0]
	}

	if config.Name == "" {
		config.Name = uniqueName("input")
	}
	if config.Sparse == "" {
		config.Sparse = TfDefault
	}

	if config.Ragged == "" {
		config.Ragged = TfDefault
	}

	return &Input{
		shape:     shape,
		batchSize: config.BatchSize,
		name:      config.Name,
		dtype:     dType,
		sparse:    config.Sparse,
		ragged:    config.Ragged,
	}
}

func (i *Input) GetShape() tf.Shape {
	return i.shape
}

func (i *Input) GetDtype() DataType {
	return i.dtype
}

func (i *Input) SetInput(inputs []Layer) {

}

func (i *Input) GetImport() string {
	return "from tensorflow.keras.layers import Input"
}

func (i *Input) GetPythonVariableName() string {
	return i.name
}

func (i *Input) GetPythonDefinitionString() string {
	// TODO: this is nasty, replace it with json configs
	args := []string{
		fmt.Sprintf(`name="%s"`, i.name),
		fmt.Sprintf("shape=%s", strings.ReplaceAll(i.shape.String(), "?", "None")),
		fmt.Sprintf("dtype=%s", i.dtype),
	}

	if i.ragged != TfDefault {
		args = append(args, fmt.Sprintf("ragged=%s", i.ragged))
	}

	if i.sparse != TfDefault {
		args = append(args, fmt.Sprintf("sparse=%s", i.sparse))
	}

	return fmt.Sprintf(
		`Input(
    %s
)`,
		strings.Join(args, ",\n    "),
	)
}

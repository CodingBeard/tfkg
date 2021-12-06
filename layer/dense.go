package layer

import (
	"fmt"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"strconv"
	"strings"
)

type Dense struct {
	units             int
	shape             tf.Shape
	name              string
	dtype             DataType
	activation        string
	useBias           TfBool
	kernelInitializer string
	biasInitializer   string
	input             Layer
}

type DenseConfig struct {
	Name              string
	Activation        string
	UseBias           TfBool
	KernelInitializer string
	BiasInitializer   string
}

func NewDense(units int, dType DataType, optionalConfig ...DenseConfig) *Dense {
	var config DenseConfig
	if len(optionalConfig) == 1 {
		config = optionalConfig[0]
	}

	if config.Name == "" {
		config.Name = uniqueName("dense")
	}

	if config.UseBias == "" {
		config.UseBias = TfTrue
	}

	if config.KernelInitializer == "" {
		config.KernelInitializer = "glorot_uniform"
	}

	if config.BiasInitializer == "" {
		config.BiasInitializer = "zeros"
	}

	return &Dense{
		units:             units,
		shape:             tf.MakeShape(-1, int64(units)),
		name:              config.Name,
		dtype:             dType,
		useBias:           config.UseBias,
		kernelInitializer: config.KernelInitializer,
		biasInitializer:   config.BiasInitializer,
		activation:        config.Activation,
	}
}

func (d *Dense) GetShape() tf.Shape {
	return d.shape
}

func (d *Dense) GetDtype() DataType {
	return d.dtype
}

func (d *Dense) SetInput(inputs []Layer) {
	d.input = inputs[0]
}

func (d *Dense) GetImport() string {
	return "from tensorflow.keras.layers import Dense"
}

func (d *Dense) GetPythonVariableName() string {
	return d.name
}

func (d *Dense) GetPythonDefinitionString() string {
	// TODO: this is nasty, replace it with json configs
	args := []string{
		strconv.Itoa(d.units),
		fmt.Sprintf(`name="%s"`, d.name),
		fmt.Sprintf("dtype=%s", d.dtype),
	}

	if d.useBias != TfTrue {
		args = append(args, fmt.Sprintf("use_bias=%s", d.useBias))
	}

	if d.kernelInitializer != "glorot_uniform" {
		args = append(args, fmt.Sprintf(`kernel_initializer="%s"`, d.kernelInitializer))
	}

	if d.biasInitializer != "zeros" {
		args = append(args, fmt.Sprintf(`bias_initializer="%s"`, d.biasInitializer))
	}

	if d.activation != "" {
		args = append(args, fmt.Sprintf(`activation="%s"`, d.activation))
	}

	return fmt.Sprintf(
		`Dense(
    %s
)(%s)`,
		strings.Join(args, ",\n    "),
		d.input.GetPythonVariableName(),
	)
}

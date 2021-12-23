package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type Conv1D struct {
	name                string
	dtype               DataType
	inputs              []Layer
	shape               tf.Shape
	trainable           bool
	filters             float64
	kernelSize          float64
	strides             float64
	padding             string
	dataFormat          string
	dilationRate        float64
	groups              float64
	activation          string
	useBias             bool
	kernelInitializer   initializer.Initializer
	biasInitializer     initializer.Initializer
	kernelRegularizer   regularizer.Regularizer
	biasRegularizer     regularizer.Regularizer
	activityRegularizer regularizer.Regularizer
	kernelConstraint    constraint.Constraint
	biasConstraint      constraint.Constraint
}

func NewConv1D(filters float64, kernelSize float64, options ...Conv1DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		c := &Conv1D{
			filters:             filters,
			kernelSize:          kernelSize,
			strides:             1,
			padding:             "valid",
			dataFormat:          "channels_last",
			dilationRate:        1,
			groups:              1,
			activation:          "linear",
			useBias:             true,
			kernelInitializer:   &initializer.GlorotUniform{},
			biasInitializer:     &initializer.Zeros{},
			kernelRegularizer:   &regularizer.NilRegularizer{},
			biasRegularizer:     &regularizer.NilRegularizer{},
			activityRegularizer: &regularizer.NilRegularizer{},
			kernelConstraint:    &constraint.NilConstraint{},
			biasConstraint:      &constraint.NilConstraint{},
			trainable:           true,
			inputs:              inputs,
			name:                UniqueName("conv1d"),
		}
		for _, option := range options {
			option(c)
		}
		return c
	}
}

type Conv1DOption func(*Conv1D)

func Conv1DWithName(name string) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.name = name
	}
}

func Conv1DWithDtype(dtype DataType) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.dtype = dtype
	}
}

func Conv1DWithTrainable(trainable bool) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.trainable = trainable
	}
}

func Conv1DWithStrides(strides float64) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.strides = strides
	}
}

func Conv1DWithPadding(padding string) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.padding = padding
	}
}

func Conv1DWithDataFormat(dataFormat string) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.dataFormat = dataFormat
	}
}

func Conv1DWithDilationRate(dilationRate float64) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.dilationRate = dilationRate
	}
}

func Conv1DWithGroups(groups float64) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.groups = groups
	}
}

func Conv1DWithActivation(activation string) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.activation = activation
	}
}

func Conv1DWithUseBias(useBias bool) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.useBias = useBias
	}
}

func Conv1DWithKernelInitializer(kernelInitializer initializer.Initializer) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.kernelInitializer = kernelInitializer
	}
}

func Conv1DWithBiasInitializer(biasInitializer initializer.Initializer) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.biasInitializer = biasInitializer
	}
}

func Conv1DWithKernelRegularizer(kernelRegularizer regularizer.Regularizer) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.kernelRegularizer = kernelRegularizer
	}
}

func Conv1DWithBiasRegularizer(biasRegularizer regularizer.Regularizer) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.biasRegularizer = biasRegularizer
	}
}

func Conv1DWithActivityRegularizer(activityRegularizer regularizer.Regularizer) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.activityRegularizer = activityRegularizer
	}
}

func Conv1DWithKernelConstraint(kernelConstraint constraint.Constraint) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.kernelConstraint = kernelConstraint
	}
}

func Conv1DWithBiasConstraint(biasConstraint constraint.Constraint) func(c *Conv1D) {
	return func(c *Conv1D) {
		c.biasConstraint = biasConstraint
	}
}

func (c *Conv1D) GetShape() tf.Shape {
	return c.shape
}

func (c *Conv1D) GetDtype() DataType {
	return c.dtype
}

func (c *Conv1D) SetInput(inputs []Layer) {
	c.inputs = inputs
	c.dtype = inputs[0].GetDtype()
}

func (c *Conv1D) GetInputs() []Layer {
	return c.inputs
}

func (c *Conv1D) GetName() string {
	return c.name
}

type jsonConfigConv1D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (c *Conv1D) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range c.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigConv1D{
		ClassName: "Conv1D",
		Name:      c.name,
		Config: map[string]interface{}{
			"activation":           c.activation,
			"activity_regularizer": c.activityRegularizer.GetKerasLayerConfig(),
			"bias_constraint":      c.biasConstraint.GetKerasLayerConfig(),
			"bias_initializer":     c.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer":     c.biasRegularizer.GetKerasLayerConfig(),
			"data_format":          c.dataFormat,
			"dilation_rate":        c.dilationRate,
			"dtype":                c.dtype.String(),
			"filters":              c.filters,
			"groups":               c.groups,
			"kernel_constraint":    c.kernelConstraint.GetKerasLayerConfig(),
			"kernel_initializer":   c.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer":   c.kernelRegularizer.GetKerasLayerConfig(),
			"kernel_size":          c.kernelSize,
			"name":                 c.name,
			"padding":              c.padding,
			"strides":              c.strides,
			"trainable":            c.trainable,
			"use_bias":             c.useBias,
		},
		InboundNodes: inboundNodes,
	}
}

func (c *Conv1D) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type ConvLSTM2D struct {
	name                 string
	dtype                DataType
	inputs               []Layer
	shape                tf.Shape
	trainable            bool
	filters              float64
	kernelSize           float64
	strides              []interface{}
	padding              string
	dataFormat           interface{}
	dilationRate         []interface{}
	activation           string
	recurrentActivation  string
	useBias              bool
	kernelInitializer    initializer.Initializer
	recurrentInitializer initializer.Initializer
	biasInitializer      initializer.Initializer
	unitForgetBias       bool
	kernelRegularizer    regularizer.Regularizer
	recurrentRegularizer regularizer.Regularizer
	biasRegularizer      regularizer.Regularizer
	activityRegularizer  regularizer.Regularizer
	kernelConstraint     constraint.Constraint
	recurrentConstraint  constraint.Constraint
	biasConstraint       constraint.Constraint
	returnSequences      bool
	returnState          bool
	goBackwards          bool
	stateful             bool
	dropout              float64
	recurrentDropout     float64
	timeMajor            bool
	unroll               bool
}

func NewConvLSTM2D(filters float64, kernelSize float64, options ...ConvLSTM2DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		c := &ConvLSTM2D{
			filters:              filters,
			kernelSize:           kernelSize,
			strides:              []interface{}{1, 1},
			padding:              "valid",
			dataFormat:           nil,
			dilationRate:         []interface{}{1, 1},
			activation:           "tanh",
			recurrentActivation:  "hard_sigmoid",
			useBias:              true,
			kernelInitializer:    &initializer.GlorotUniform{},
			recurrentInitializer: &initializer.Orthogonal{},
			biasInitializer:      &initializer.Zeros{},
			unitForgetBias:       true,
			kernelRegularizer:    &regularizer.NilRegularizer{},
			recurrentRegularizer: &regularizer.NilRegularizer{},
			biasRegularizer:      &regularizer.NilRegularizer{},
			activityRegularizer:  &regularizer.NilRegularizer{},
			kernelConstraint:     &constraint.NilConstraint{},
			recurrentConstraint:  &constraint.NilConstraint{},
			biasConstraint:       &constraint.NilConstraint{},
			returnSequences:      false,
			returnState:          false,
			goBackwards:          false,
			stateful:             false,
			dropout:              0,
			recurrentDropout:     0,
			timeMajor:            false,
			unroll:               false,
			trainable:            true,
			inputs:               inputs,
			name:                 UniqueName("convlstm2d"),
		}
		for _, option := range options {
			option(c)
		}
		return c
	}
}

type ConvLSTM2DOption func(*ConvLSTM2D)

func ConvLSTM2DWithName(name string) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.name = name
	}
}

func ConvLSTM2DWithDtype(dtype DataType) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.dtype = dtype
	}
}

func ConvLSTM2DWithTrainable(trainable bool) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.trainable = trainable
	}
}

func ConvLSTM2DWithStrides(strides []interface{}) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.strides = strides
	}
}

func ConvLSTM2DWithPadding(padding string) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.padding = padding
	}
}

func ConvLSTM2DWithDataFormat(dataFormat interface{}) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.dataFormat = dataFormat
	}
}

func ConvLSTM2DWithDilationRate(dilationRate []interface{}) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.dilationRate = dilationRate
	}
}

func ConvLSTM2DWithActivation(activation string) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.activation = activation
	}
}

func ConvLSTM2DWithRecurrentActivation(recurrentActivation string) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.recurrentActivation = recurrentActivation
	}
}

func ConvLSTM2DWithUseBias(useBias bool) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.useBias = useBias
	}
}

func ConvLSTM2DWithKernelInitializer(kernelInitializer initializer.Initializer) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.kernelInitializer = kernelInitializer
	}
}

func ConvLSTM2DWithRecurrentInitializer(recurrentInitializer initializer.Initializer) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.recurrentInitializer = recurrentInitializer
	}
}

func ConvLSTM2DWithBiasInitializer(biasInitializer initializer.Initializer) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.biasInitializer = biasInitializer
	}
}

func ConvLSTM2DWithUnitForgetBias(unitForgetBias bool) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.unitForgetBias = unitForgetBias
	}
}

func ConvLSTM2DWithKernelRegularizer(kernelRegularizer regularizer.Regularizer) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.kernelRegularizer = kernelRegularizer
	}
}

func ConvLSTM2DWithRecurrentRegularizer(recurrentRegularizer regularizer.Regularizer) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.recurrentRegularizer = recurrentRegularizer
	}
}

func ConvLSTM2DWithBiasRegularizer(biasRegularizer regularizer.Regularizer) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.biasRegularizer = biasRegularizer
	}
}

func ConvLSTM2DWithActivityRegularizer(activityRegularizer regularizer.Regularizer) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.activityRegularizer = activityRegularizer
	}
}

func ConvLSTM2DWithKernelConstraint(kernelConstraint constraint.Constraint) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.kernelConstraint = kernelConstraint
	}
}

func ConvLSTM2DWithRecurrentConstraint(recurrentConstraint constraint.Constraint) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.recurrentConstraint = recurrentConstraint
	}
}

func ConvLSTM2DWithBiasConstraint(biasConstraint constraint.Constraint) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.biasConstraint = biasConstraint
	}
}

func ConvLSTM2DWithReturnSequences(returnSequences bool) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.returnSequences = returnSequences
	}
}

func ConvLSTM2DWithReturnState(returnState bool) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.returnState = returnState
	}
}

func ConvLSTM2DWithGoBackwards(goBackwards bool) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.goBackwards = goBackwards
	}
}

func ConvLSTM2DWithStateful(stateful bool) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.stateful = stateful
	}
}

func ConvLSTM2DWithDropout(dropout float64) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.dropout = dropout
	}
}

func ConvLSTM2DWithRecurrentDropout(recurrentDropout float64) func(c *ConvLSTM2D) {
	return func(c *ConvLSTM2D) {
		c.recurrentDropout = recurrentDropout
	}
}

func (c *ConvLSTM2D) GetShape() tf.Shape {
	return c.shape
}

func (c *ConvLSTM2D) GetDtype() DataType {
	return c.dtype
}

func (c *ConvLSTM2D) SetInput(inputs []Layer) {
	c.inputs = inputs
	c.dtype = inputs[0].GetDtype()
}

func (c *ConvLSTM2D) GetInputs() []Layer {
	return c.inputs
}

func (c *ConvLSTM2D) GetName() string {
	return c.name
}

type jsonConfigConvLSTM2D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (c *ConvLSTM2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigConvLSTM2D{
		ClassName: "ConvLSTM2D",
		Name:      c.name,
		Config: map[string]interface{}{
			"activation":            c.activation,
			"activity_regularizer":  c.activityRegularizer.GetKerasLayerConfig(),
			"bias_constraint":       c.biasConstraint.GetKerasLayerConfig(),
			"bias_initializer":      c.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer":      c.biasRegularizer.GetKerasLayerConfig(),
			"data_format":           c.dataFormat,
			"dilation_rate":         c.dilationRate,
			"dropout":               c.dropout,
			"dtype":                 c.dtype.String(),
			"filters":               c.filters,
			"go_backwards":          c.goBackwards,
			"kernel_constraint":     c.kernelConstraint.GetKerasLayerConfig(),
			"kernel_initializer":    c.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer":    c.kernelRegularizer.GetKerasLayerConfig(),
			"kernel_size":           c.kernelSize,
			"name":                  c.name,
			"padding":               c.padding,
			"recurrent_activation":  c.recurrentActivation,
			"recurrent_constraint":  c.recurrentConstraint.GetKerasLayerConfig(),
			"recurrent_dropout":     c.recurrentDropout,
			"recurrent_initializer": c.recurrentInitializer.GetKerasLayerConfig(),
			"recurrent_regularizer": c.recurrentRegularizer.GetKerasLayerConfig(),
			"return_sequences":      c.returnSequences,
			"return_state":          c.returnState,
			"stateful":              c.stateful,
			"strides":               c.strides,
			"time_major":            c.timeMajor,
			"trainable":             c.trainable,
			"unit_forget_bias":      c.unitForgetBias,
			"unroll":                c.unroll,
			"use_bias":              c.useBias,
		},
		InboundNodes: inboundNodes,
	}
}

func (c *ConvLSTM2D) GetCustomLayerDefinition() string {
	return ``
}

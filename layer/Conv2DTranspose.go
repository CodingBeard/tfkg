package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type Conv2DTranspose struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	filters float64
	kernelSize float64
	strides []interface {}
	padding string
	outputPadding interface{}
	dataFormat interface{}
	dilationRate []interface {}
	activation string
	useBias bool
	kernelInitializer initializer.Initializer
	biasInitializer initializer.Initializer
	kernelRegularizer regularizer.Regularizer
	biasRegularizer regularizer.Regularizer
	activityRegularizer regularizer.Regularizer
	kernelConstraint constraint.Constraint
	biasConstraint constraint.Constraint
	groups float64
}

func NewConv2DTranspose(filters float64, kernelSize float64, options ...Conv2DTransposeOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		c := &Conv2DTranspose{
			filters: filters,
			kernelSize: kernelSize,
			strides: []interface {}{1, 1},
			padding: "valid",
			outputPadding: nil,
			dataFormat: nil,
			dilationRate: []interface {}{1, 1},
			activation: "linear",
			useBias: true,
			kernelInitializer: &initializer.GlorotUniform{},
			biasInitializer: &initializer.Zeros{},
			kernelRegularizer: &regularizer.NilRegularizer{},
			biasRegularizer: &regularizer.NilRegularizer{},
			activityRegularizer: &regularizer.NilRegularizer{},
			kernelConstraint: &constraint.NilConstraint{},
			biasConstraint: &constraint.NilConstraint{},
			groups: 1,
			trainable: true,
			inputs: inputs,
			name: uniqueName("conv2dtranspose"),		
		}
		for _, option := range options {
			option(c)
		}
		return c
	}
}

type Conv2DTransposeOption func (*Conv2DTranspose)

func Conv2DTransposeWithName(name string) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.name = name
	}
}

func Conv2DTransposeWithDtype(dtype DataType) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.dtype = dtype
	}
}

func Conv2DTransposeWithTrainable(trainable bool) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.trainable = trainable
	}
}

func Conv2DTransposeWithStrides(strides []interface {}) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.strides = strides
	}
}

func Conv2DTransposeWithPadding(padding string) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.padding = padding
	}
}

func Conv2DTransposeWithOutputPadding(outputPadding interface{}) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.outputPadding = outputPadding
	}
}

func Conv2DTransposeWithDataFormat(dataFormat interface{}) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.dataFormat = dataFormat
	}
}

func Conv2DTransposeWithDilationRate(dilationRate []interface {}) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.dilationRate = dilationRate
	}
}

func Conv2DTransposeWithActivation(activation string) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.activation = activation
	}
}

func Conv2DTransposeWithUseBias(useBias bool) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.useBias = useBias
	}
}

func Conv2DTransposeWithKernelInitializer(kernelInitializer initializer.Initializer) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.kernelInitializer = kernelInitializer
	}
}

func Conv2DTransposeWithBiasInitializer(biasInitializer initializer.Initializer) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.biasInitializer = biasInitializer
	}
}

func Conv2DTransposeWithKernelRegularizer(kernelRegularizer regularizer.Regularizer) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.kernelRegularizer = kernelRegularizer
	}
}

func Conv2DTransposeWithBiasRegularizer(biasRegularizer regularizer.Regularizer) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.biasRegularizer = biasRegularizer
	}
}

func Conv2DTransposeWithActivityRegularizer(activityRegularizer regularizer.Regularizer) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.activityRegularizer = activityRegularizer
	}
}

func Conv2DTransposeWithKernelConstraint(kernelConstraint constraint.Constraint) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.kernelConstraint = kernelConstraint
	}
}

func Conv2DTransposeWithBiasConstraint(biasConstraint constraint.Constraint) func(c *Conv2DTranspose) {
	 return func(c *Conv2DTranspose) {
		c.biasConstraint = biasConstraint
	}
}


func (c *Conv2DTranspose) GetShape() tf.Shape {
	return c.shape
}

func (c *Conv2DTranspose) GetDtype() DataType {
	return c.dtype
}

func (c *Conv2DTranspose) SetInput(inputs []Layer) {
	c.inputs = inputs
	c.dtype = inputs[0].GetDtype()
}

func (c *Conv2DTranspose) GetInputs() []Layer {
	return c.inputs
}

func (c *Conv2DTranspose) GetName() string {
	return c.name
}


type jsonConfigConv2DTranspose struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (c *Conv2DTranspose) GetKerasLayerConfig() interface{} {
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
	return jsonConfigConv2DTranspose{
		ClassName: "Conv2DTranspose",
		Name: c.name,
		Config: map[string]interface{}{
			"padding": c.padding,
			"dilation_rate": c.dilationRate,
			"kernel_initializer": c.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer": c.kernelRegularizer.GetKerasLayerConfig(),
			"output_padding": c.outputPadding,
			"name": c.name,
			"kernel_size": c.kernelSize,
			"bias_regularizer": c.biasRegularizer.GetKerasLayerConfig(),
			"kernel_constraint": c.kernelConstraint.GetKerasLayerConfig(),
			"bias_constraint": c.biasConstraint.GetKerasLayerConfig(),
			"strides": c.strides,
			"data_format": c.dataFormat,
			"activation": c.activation,
			"bias_initializer": c.biasInitializer.GetKerasLayerConfig(),
			"activity_regularizer": c.activityRegularizer.GetKerasLayerConfig(),
			"dtype": c.dtype.String(),
			"groups": c.groups,
			"use_bias": c.useBias,
			"trainable": c.trainable,
			"filters": c.filters,
		},
		InboundNodes: inboundNodes,
	}
}
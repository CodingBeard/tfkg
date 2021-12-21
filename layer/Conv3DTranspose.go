package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type Conv3DTranspose struct {
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

func NewConv3DTranspose(filters float64, kernelSize float64, options ...Conv3DTransposeOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		c := &Conv3DTranspose{
			filters: filters,
			kernelSize: kernelSize,
			strides: []interface {}{1, 1, 1},
			padding: "valid",
			outputPadding: nil,
			dataFormat: nil,
			dilationRate: []interface {}{1, 1, 1},
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
			name: uniqueName("conv3dtranspose"),		
		}
		for _, option := range options {
			option(c)
		}
		return c
	}
}

type Conv3DTransposeOption func (*Conv3DTranspose)

func Conv3DTransposeWithName(name string) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.name = name
	}
}

func Conv3DTransposeWithDtype(dtype DataType) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.dtype = dtype
	}
}

func Conv3DTransposeWithTrainable(trainable bool) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.trainable = trainable
	}
}

func Conv3DTransposeWithStrides(strides []interface {}) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.strides = strides
	}
}

func Conv3DTransposeWithPadding(padding string) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.padding = padding
	}
}

func Conv3DTransposeWithOutputPadding(outputPadding interface{}) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.outputPadding = outputPadding
	}
}

func Conv3DTransposeWithDataFormat(dataFormat interface{}) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.dataFormat = dataFormat
	}
}

func Conv3DTransposeWithDilationRate(dilationRate []interface {}) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.dilationRate = dilationRate
	}
}

func Conv3DTransposeWithActivation(activation string) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.activation = activation
	}
}

func Conv3DTransposeWithUseBias(useBias bool) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.useBias = useBias
	}
}

func Conv3DTransposeWithKernelInitializer(kernelInitializer initializer.Initializer) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.kernelInitializer = kernelInitializer
	}
}

func Conv3DTransposeWithBiasInitializer(biasInitializer initializer.Initializer) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.biasInitializer = biasInitializer
	}
}

func Conv3DTransposeWithKernelRegularizer(kernelRegularizer regularizer.Regularizer) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.kernelRegularizer = kernelRegularizer
	}
}

func Conv3DTransposeWithBiasRegularizer(biasRegularizer regularizer.Regularizer) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.biasRegularizer = biasRegularizer
	}
}

func Conv3DTransposeWithActivityRegularizer(activityRegularizer regularizer.Regularizer) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.activityRegularizer = activityRegularizer
	}
}

func Conv3DTransposeWithKernelConstraint(kernelConstraint constraint.Constraint) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.kernelConstraint = kernelConstraint
	}
}

func Conv3DTransposeWithBiasConstraint(biasConstraint constraint.Constraint) func(c *Conv3DTranspose) {
	 return func(c *Conv3DTranspose) {
		c.biasConstraint = biasConstraint
	}
}


func (c *Conv3DTranspose) GetShape() tf.Shape {
	return c.shape
}

func (c *Conv3DTranspose) GetDtype() DataType {
	return c.dtype
}

func (c *Conv3DTranspose) SetInput(inputs []Layer) {
	c.inputs = inputs
	c.dtype = inputs[0].GetDtype()
}

func (c *Conv3DTranspose) GetInputs() []Layer {
	return c.inputs
}

func (c *Conv3DTranspose) GetName() string {
	return c.name
}


type jsonConfigConv3DTranspose struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (c *Conv3DTranspose) GetKerasLayerConfig() interface{} {
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
	return jsonConfigConv3DTranspose{
		ClassName: "Conv3DTranspose",
		Name: c.name,
		Config: map[string]interface{}{
			"strides": c.strides,
			"use_bias": c.useBias,
			"kernel_size": c.kernelSize,
			"padding": c.padding,
			"bias_initializer": c.biasInitializer.GetKerasLayerConfig(),
			"kernel_regularizer": c.kernelRegularizer.GetKerasLayerConfig(),
			"activity_regularizer": c.activityRegularizer.GetKerasLayerConfig(),
			"kernel_constraint": c.kernelConstraint.GetKerasLayerConfig(),
			"name": c.name,
			"trainable": c.trainable,
			"filters": c.filters,
			"data_format": c.dataFormat,
			"groups": c.groups,
			"activation": c.activation,
			"kernel_initializer": c.kernelInitializer.GetKerasLayerConfig(),
			"bias_regularizer": c.biasRegularizer.GetKerasLayerConfig(),
			"dtype": c.dtype.String(),
			"bias_constraint": c.biasConstraint.GetKerasLayerConfig(),
			"output_padding": c.outputPadding,
		},
		InboundNodes: inboundNodes,
	}
}
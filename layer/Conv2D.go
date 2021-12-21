package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type Conv2D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	filters float64
	kernelSize float64
	strides []interface {}
	padding string
	dataFormat interface{}
	dilationRate []interface {}
	groups float64
	activation string
	useBias bool
	kernelInitializer initializer.Initializer
	biasInitializer initializer.Initializer
	kernelRegularizer regularizer.Regularizer
	biasRegularizer regularizer.Regularizer
	activityRegularizer regularizer.Regularizer
	kernelConstraint constraint.Constraint
	biasConstraint constraint.Constraint
}

func NewConv2D(filters float64, kernelSize float64, options ...Conv2DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		c := &Conv2D{
			filters: filters,
			kernelSize: kernelSize,
			strides: []interface {}{1, 1},
			padding: "valid",
			dataFormat: nil,
			dilationRate: []interface {}{1, 1},
			groups: 1,
			activation: "linear",
			useBias: true,
			kernelInitializer: &initializer.GlorotUniform{},
			biasInitializer: &initializer.Zeros{},
			kernelRegularizer: &regularizer.NilRegularizer{},
			biasRegularizer: &regularizer.NilRegularizer{},
			activityRegularizer: &regularizer.NilRegularizer{},
			kernelConstraint: &constraint.NilConstraint{},
			biasConstraint: &constraint.NilConstraint{},
			trainable: true,
			inputs: inputs,
			name: uniqueName("conv2d"),		
		}
		for _, option := range options {
			option(c)
		}
		return c
	}
}

type Conv2DOption func (*Conv2D)

func Conv2DWithName(name string) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.name = name
	}
}

func Conv2DWithDtype(dtype DataType) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.dtype = dtype
	}
}

func Conv2DWithTrainable(trainable bool) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.trainable = trainable
	}
}

func Conv2DWithStrides(strides []interface {}) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.strides = strides
	}
}

func Conv2DWithPadding(padding string) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.padding = padding
	}
}

func Conv2DWithDataFormat(dataFormat interface{}) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.dataFormat = dataFormat
	}
}

func Conv2DWithDilationRate(dilationRate []interface {}) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.dilationRate = dilationRate
	}
}

func Conv2DWithGroups(groups float64) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.groups = groups
	}
}

func Conv2DWithActivation(activation string) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.activation = activation
	}
}

func Conv2DWithUseBias(useBias bool) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.useBias = useBias
	}
}

func Conv2DWithKernelInitializer(kernelInitializer initializer.Initializer) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.kernelInitializer = kernelInitializer
	}
}

func Conv2DWithBiasInitializer(biasInitializer initializer.Initializer) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.biasInitializer = biasInitializer
	}
}

func Conv2DWithKernelRegularizer(kernelRegularizer regularizer.Regularizer) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.kernelRegularizer = kernelRegularizer
	}
}

func Conv2DWithBiasRegularizer(biasRegularizer regularizer.Regularizer) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.biasRegularizer = biasRegularizer
	}
}

func Conv2DWithActivityRegularizer(activityRegularizer regularizer.Regularizer) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.activityRegularizer = activityRegularizer
	}
}

func Conv2DWithKernelConstraint(kernelConstraint constraint.Constraint) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.kernelConstraint = kernelConstraint
	}
}

func Conv2DWithBiasConstraint(biasConstraint constraint.Constraint) func(c *Conv2D) {
	 return func(c *Conv2D) {
		c.biasConstraint = biasConstraint
	}
}


func (c *Conv2D) GetShape() tf.Shape {
	return c.shape
}

func (c *Conv2D) GetDtype() DataType {
	return c.dtype
}

func (c *Conv2D) SetInput(inputs []Layer) {
	c.inputs = inputs
	c.dtype = inputs[0].GetDtype()
}

func (c *Conv2D) GetInputs() []Layer {
	return c.inputs
}

func (c *Conv2D) GetName() string {
	return c.name
}


type jsonConfigConv2D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (c *Conv2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigConv2D{
		ClassName: "Conv2D",
		Name: c.name,
		Config: map[string]interface{}{
			"kernel_constraint": c.kernelConstraint.GetKerasLayerConfig(),
			"kernel_size": c.kernelSize,
			"data_format": c.dataFormat,
			"activation": c.activation,
			"use_bias": c.useBias,
			"activity_regularizer": c.activityRegularizer.GetKerasLayerConfig(),
			"name": c.name,
			"trainable": c.trainable,
			"strides": c.strides,
			"bias_constraint": c.biasConstraint.GetKerasLayerConfig(),
			"dtype": c.dtype.String(),
			"dilation_rate": c.dilationRate,
			"groups": c.groups,
			"bias_initializer": c.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer": c.biasRegularizer.GetKerasLayerConfig(),
			"filters": c.filters,
			"padding": c.padding,
			"kernel_initializer": c.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer": c.kernelRegularizer.GetKerasLayerConfig(),
		},
		InboundNodes: inboundNodes,
	}
}
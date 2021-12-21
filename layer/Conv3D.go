package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type Conv3D struct {
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

func NewConv3D(filters float64, kernelSize float64, options ...Conv3DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		c := &Conv3D{
			filters: filters,
			kernelSize: kernelSize,
			strides: []interface {}{1, 1, 1},
			padding: "valid",
			dataFormat: nil,
			dilationRate: []interface {}{1, 1, 1},
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
			name: uniqueName("conv3d"),		
		}
		for _, option := range options {
			option(c)
		}
		return c
	}
}

type Conv3DOption func (*Conv3D)

func Conv3DWithName(name string) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.name = name
	}
}

func Conv3DWithDtype(dtype DataType) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.dtype = dtype
	}
}

func Conv3DWithTrainable(trainable bool) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.trainable = trainable
	}
}

func Conv3DWithStrides(strides []interface {}) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.strides = strides
	}
}

func Conv3DWithPadding(padding string) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.padding = padding
	}
}

func Conv3DWithDataFormat(dataFormat interface{}) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.dataFormat = dataFormat
	}
}

func Conv3DWithDilationRate(dilationRate []interface {}) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.dilationRate = dilationRate
	}
}

func Conv3DWithGroups(groups float64) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.groups = groups
	}
}

func Conv3DWithActivation(activation string) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.activation = activation
	}
}

func Conv3DWithUseBias(useBias bool) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.useBias = useBias
	}
}

func Conv3DWithKernelInitializer(kernelInitializer initializer.Initializer) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.kernelInitializer = kernelInitializer
	}
}

func Conv3DWithBiasInitializer(biasInitializer initializer.Initializer) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.biasInitializer = biasInitializer
	}
}

func Conv3DWithKernelRegularizer(kernelRegularizer regularizer.Regularizer) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.kernelRegularizer = kernelRegularizer
	}
}

func Conv3DWithBiasRegularizer(biasRegularizer regularizer.Regularizer) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.biasRegularizer = biasRegularizer
	}
}

func Conv3DWithActivityRegularizer(activityRegularizer regularizer.Regularizer) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.activityRegularizer = activityRegularizer
	}
}

func Conv3DWithKernelConstraint(kernelConstraint constraint.Constraint) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.kernelConstraint = kernelConstraint
	}
}

func Conv3DWithBiasConstraint(biasConstraint constraint.Constraint) func(c *Conv3D) {
	 return func(c *Conv3D) {
		c.biasConstraint = biasConstraint
	}
}


func (c *Conv3D) GetShape() tf.Shape {
	return c.shape
}

func (c *Conv3D) GetDtype() DataType {
	return c.dtype
}

func (c *Conv3D) SetInput(inputs []Layer) {
	c.inputs = inputs
	c.dtype = inputs[0].GetDtype()
}

func (c *Conv3D) GetInputs() []Layer {
	return c.inputs
}

func (c *Conv3D) GetName() string {
	return c.name
}


type jsonConfigConv3D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (c *Conv3D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigConv3D{
		ClassName: "Conv3D",
		Name: c.name,
		Config: map[string]interface{}{
			"kernel_size": c.kernelSize,
			"use_bias": c.useBias,
			"bias_regularizer": c.biasRegularizer.GetKerasLayerConfig(),
			"bias_constraint": c.biasConstraint.GetKerasLayerConfig(),
			"name": c.name,
			"filters": c.filters,
			"padding": c.padding,
			"kernel_initializer": c.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer": c.kernelRegularizer.GetKerasLayerConfig(),
			"activity_regularizer": c.activityRegularizer.GetKerasLayerConfig(),
			"kernel_constraint": c.kernelConstraint.GetKerasLayerConfig(),
			"trainable": c.trainable,
			"strides": c.strides,
			"data_format": c.dataFormat,
			"dilation_rate": c.dilationRate,
			"activation": c.activation,
			"bias_initializer": c.biasInitializer.GetKerasLayerConfig(),
			"dtype": c.dtype.String(),
			"groups": c.groups,
		},
		InboundNodes: inboundNodes,
	}
}
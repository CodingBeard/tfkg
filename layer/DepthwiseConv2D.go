package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type DepthwiseConv2D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	kernelSize float64
	strides []interface {}
	padding string
	depthMultiplier float64
	dataFormat interface{}
	dilationRate []interface {}
	activation string
	useBias bool
	depthwiseInitializer initializer.Initializer
	biasInitializer initializer.Initializer
	depthwiseRegularizer regularizer.Regularizer
	biasRegularizer regularizer.Regularizer
	activityRegularizer regularizer.Regularizer
	depthwiseConstraint constraint.Constraint
	biasConstraint constraint.Constraint
	groups float64
}

func NewDepthwiseConv2D(kernelSize float64, options ...DepthwiseConv2DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		d := &DepthwiseConv2D{
			kernelSize: kernelSize,
			strides: []interface {}{1, 1},
			padding: "valid",
			depthMultiplier: 1,
			dataFormat: nil,
			dilationRate: []interface {}{1, 1},
			activation: "linear",
			useBias: true,
			depthwiseInitializer: &initializer.GlorotUniform{},
			biasInitializer: &initializer.Zeros{},
			depthwiseRegularizer: &regularizer.NilRegularizer{},
			biasRegularizer: &regularizer.NilRegularizer{},
			activityRegularizer: &regularizer.NilRegularizer{},
			depthwiseConstraint: &constraint.NilConstraint{},
			biasConstraint: &constraint.NilConstraint{},
			groups: 1,
			trainable: true,
			inputs: inputs,
			name: uniqueName("depthwiseconv2d"),		
		}
		for _, option := range options {
			option(d)
		}
		return d
	}
}

type DepthwiseConv2DOption func (*DepthwiseConv2D)

func DepthwiseConv2DWithName(name string) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.name = name
	}
}

func DepthwiseConv2DWithDtype(dtype DataType) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.dtype = dtype
	}
}

func DepthwiseConv2DWithTrainable(trainable bool) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.trainable = trainable
	}
}

func DepthwiseConv2DWithStrides(strides []interface {}) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.strides = strides
	}
}

func DepthwiseConv2DWithPadding(padding string) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.padding = padding
	}
}

func DepthwiseConv2DWithDepthMultiplier(depthMultiplier float64) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.depthMultiplier = depthMultiplier
	}
}

func DepthwiseConv2DWithDataFormat(dataFormat interface{}) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.dataFormat = dataFormat
	}
}

func DepthwiseConv2DWithDilationRate(dilationRate []interface {}) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.dilationRate = dilationRate
	}
}

func DepthwiseConv2DWithActivation(activation string) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.activation = activation
	}
}

func DepthwiseConv2DWithUseBias(useBias bool) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.useBias = useBias
	}
}

func DepthwiseConv2DWithDepthwiseInitializer(depthwiseInitializer initializer.Initializer) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.depthwiseInitializer = depthwiseInitializer
	}
}

func DepthwiseConv2DWithBiasInitializer(biasInitializer initializer.Initializer) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.biasInitializer = biasInitializer
	}
}

func DepthwiseConv2DWithDepthwiseRegularizer(depthwiseRegularizer regularizer.Regularizer) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.depthwiseRegularizer = depthwiseRegularizer
	}
}

func DepthwiseConv2DWithBiasRegularizer(biasRegularizer regularizer.Regularizer) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.biasRegularizer = biasRegularizer
	}
}

func DepthwiseConv2DWithActivityRegularizer(activityRegularizer regularizer.Regularizer) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.activityRegularizer = activityRegularizer
	}
}

func DepthwiseConv2DWithDepthwiseConstraint(depthwiseConstraint constraint.Constraint) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.depthwiseConstraint = depthwiseConstraint
	}
}

func DepthwiseConv2DWithBiasConstraint(biasConstraint constraint.Constraint) func(d *DepthwiseConv2D) {
	 return func(d *DepthwiseConv2D) {
		d.biasConstraint = biasConstraint
	}
}


func (d *DepthwiseConv2D) GetShape() tf.Shape {
	return d.shape
}

func (d *DepthwiseConv2D) GetDtype() DataType {
	return d.dtype
}

func (d *DepthwiseConv2D) SetInput(inputs []Layer) {
	d.inputs = inputs
	d.dtype = inputs[0].GetDtype()
}

func (d *DepthwiseConv2D) GetInputs() []Layer {
	return d.inputs
}

func (d *DepthwiseConv2D) GetName() string {
	return d.name
}


type jsonConfigDepthwiseConv2D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (d *DepthwiseConv2D) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range d.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigDepthwiseConv2D{
		ClassName: "DepthwiseConv2D",
		Name: d.name,
		Config: map[string]interface{}{
			"use_bias": d.useBias,
			"bias_regularizer": d.biasRegularizer.GetKerasLayerConfig(),
			"activity_regularizer": d.activityRegularizer.GetKerasLayerConfig(),
			"dtype": d.dtype.String(),
			"padding": d.padding,
			"activation": d.activation,
			"bias_initializer": d.biasInitializer.GetKerasLayerConfig(),
			"bias_constraint": d.biasConstraint.GetKerasLayerConfig(),
			"depth_multiplier": d.depthMultiplier,
			"depthwise_regularizer": d.depthwiseRegularizer.GetKerasLayerConfig(),
			"depthwise_constraint": d.depthwiseConstraint.GetKerasLayerConfig(),
			"trainable": d.trainable,
			"data_format": d.dataFormat,
			"dilation_rate": d.dilationRate,
			"groups": d.groups,
			"depthwise_initializer": d.depthwiseInitializer.GetKerasLayerConfig(),
			"name": d.name,
			"strides": d.strides,
			"kernel_size": d.kernelSize,
		},
		InboundNodes: inboundNodes,
	}
}
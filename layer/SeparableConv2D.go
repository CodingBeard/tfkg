package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type SeparableConv2D struct {
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
	depthMultiplier float64
	activation string
	useBias bool
	depthwiseInitializer initializer.Initializer
	pointwiseInitializer initializer.Initializer
	biasInitializer initializer.Initializer
	depthwiseRegularizer regularizer.Regularizer
	pointwiseRegularizer regularizer.Regularizer
	biasRegularizer regularizer.Regularizer
	activityRegularizer regularizer.Regularizer
	depthwiseConstraint constraint.Constraint
	pointwiseConstraint constraint.Constraint
	biasConstraint constraint.Constraint
	groups float64
	kernelConstraint constraint.Constraint
	kernelInitializer initializer.Initializer
	kernelRegularizer regularizer.Regularizer
}

func NewSeparableConv2D(filters float64, kernelSize float64, options ...SeparableConv2DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		s := &SeparableConv2D{
			filters: filters,
			kernelSize: kernelSize,
			strides: []interface {}{1, 1},
			padding: "valid",
			dataFormat: nil,
			dilationRate: []interface {}{1, 1},
			depthMultiplier: 1,
			activation: "linear",
			useBias: true,
			depthwiseInitializer: &initializer.GlorotUniform{},
			pointwiseInitializer: &initializer.GlorotUniform{},
			biasInitializer: &initializer.Zeros{},
			depthwiseRegularizer: &regularizer.NilRegularizer{},
			pointwiseRegularizer: &regularizer.NilRegularizer{},
			biasRegularizer: &regularizer.NilRegularizer{},
			activityRegularizer: &regularizer.NilRegularizer{},
			depthwiseConstraint: &constraint.NilConstraint{},
			pointwiseConstraint: &constraint.NilConstraint{},
			biasConstraint: &constraint.NilConstraint{},
			kernelRegularizer: &regularizer.NilRegularizer{},
			kernelConstraint: &constraint.NilConstraint{},
			groups: 1,
			kernelInitializer: &initializer.GlorotUniform{},
			trainable: true,
			inputs: inputs,
			name: uniqueName("separableconv2d"),		
		}
		for _, option := range options {
			option(s)
		}
		return s
	}
}

type SeparableConv2DOption func (*SeparableConv2D)

func SeparableConv2DWithName(name string) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.name = name
	}
}

func SeparableConv2DWithDtype(dtype DataType) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.dtype = dtype
	}
}

func SeparableConv2DWithTrainable(trainable bool) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.trainable = trainable
	}
}

func SeparableConv2DWithStrides(strides []interface {}) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.strides = strides
	}
}

func SeparableConv2DWithPadding(padding string) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.padding = padding
	}
}

func SeparableConv2DWithDataFormat(dataFormat interface{}) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.dataFormat = dataFormat
	}
}

func SeparableConv2DWithDilationRate(dilationRate []interface {}) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.dilationRate = dilationRate
	}
}

func SeparableConv2DWithDepthMultiplier(depthMultiplier float64) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.depthMultiplier = depthMultiplier
	}
}

func SeparableConv2DWithActivation(activation string) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.activation = activation
	}
}

func SeparableConv2DWithUseBias(useBias bool) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.useBias = useBias
	}
}

func SeparableConv2DWithDepthwiseInitializer(depthwiseInitializer initializer.Initializer) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.depthwiseInitializer = depthwiseInitializer
	}
}

func SeparableConv2DWithPointwiseInitializer(pointwiseInitializer initializer.Initializer) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.pointwiseInitializer = pointwiseInitializer
	}
}

func SeparableConv2DWithBiasInitializer(biasInitializer initializer.Initializer) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.biasInitializer = biasInitializer
	}
}

func SeparableConv2DWithDepthwiseRegularizer(depthwiseRegularizer regularizer.Regularizer) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.depthwiseRegularizer = depthwiseRegularizer
	}
}

func SeparableConv2DWithPointwiseRegularizer(pointwiseRegularizer regularizer.Regularizer) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.pointwiseRegularizer = pointwiseRegularizer
	}
}

func SeparableConv2DWithBiasRegularizer(biasRegularizer regularizer.Regularizer) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.biasRegularizer = biasRegularizer
	}
}

func SeparableConv2DWithActivityRegularizer(activityRegularizer regularizer.Regularizer) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.activityRegularizer = activityRegularizer
	}
}

func SeparableConv2DWithDepthwiseConstraint(depthwiseConstraint constraint.Constraint) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.depthwiseConstraint = depthwiseConstraint
	}
}

func SeparableConv2DWithPointwiseConstraint(pointwiseConstraint constraint.Constraint) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.pointwiseConstraint = pointwiseConstraint
	}
}

func SeparableConv2DWithBiasConstraint(biasConstraint constraint.Constraint) func(s *SeparableConv2D) {
	 return func(s *SeparableConv2D) {
		s.biasConstraint = biasConstraint
	}
}


func (s *SeparableConv2D) GetShape() tf.Shape {
	return s.shape
}

func (s *SeparableConv2D) GetDtype() DataType {
	return s.dtype
}

func (s *SeparableConv2D) SetInput(inputs []Layer) {
	s.inputs = inputs
	s.dtype = inputs[0].GetDtype()
}

func (s *SeparableConv2D) GetInputs() []Layer {
	return s.inputs
}

func (s *SeparableConv2D) GetName() string {
	return s.name
}


type jsonConfigSeparableConv2D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (s *SeparableConv2D) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range s.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigSeparableConv2D{
		ClassName: "SeparableConv2D",
		Name: s.name,
		Config: map[string]interface{}{
			"activity_regularizer": s.activityRegularizer.GetKerasLayerConfig(),
			"pointwise_constraint": s.pointwiseConstraint.GetKerasLayerConfig(),
			"trainable": s.trainable,
			"filters": s.filters,
			"use_bias": s.useBias,
			"bias_initializer": s.biasInitializer.GetKerasLayerConfig(),
			"groups": s.groups,
			"activation": s.activation,
			"kernel_constraint": s.kernelConstraint.GetKerasLayerConfig(),
			"bias_constraint": s.biasConstraint.GetKerasLayerConfig(),
			"kernel_size": s.kernelSize,
			"strides": s.strides,
			"padding": s.padding,
			"data_format": s.dataFormat,
			"depthwise_initializer": s.depthwiseInitializer.GetKerasLayerConfig(),
			"pointwise_initializer": s.pointwiseInitializer.GetKerasLayerConfig(),
			"depthwise_constraint": s.depthwiseConstraint.GetKerasLayerConfig(),
			"depth_multiplier": s.depthMultiplier,
			"pointwise_regularizer": s.pointwiseRegularizer.GetKerasLayerConfig(),
			"dtype": s.dtype.String(),
			"dilation_rate": s.dilationRate,
			"kernel_initializer": s.kernelInitializer.GetKerasLayerConfig(),
			"bias_regularizer": s.biasRegularizer.GetKerasLayerConfig(),
			"name": s.name,
			"kernel_regularizer": s.kernelRegularizer.GetKerasLayerConfig(),
			"depthwise_regularizer": s.depthwiseRegularizer.GetKerasLayerConfig(),
		},
		InboundNodes: inboundNodes,
	}
}
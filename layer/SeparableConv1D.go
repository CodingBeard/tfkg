package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type SeparableConv1D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	filters float64
	kernelSize float64
	strides float64
	padding string
	dataFormat interface{}
	dilationRate float64
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
	kernelRegularizer regularizer.Regularizer
	kernelInitializer initializer.Initializer
	kernelConstraint constraint.Constraint
	groups float64
}

func NewSeparableConv1D(filters float64, kernelSize float64, options ...SeparableConv1DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		s := &SeparableConv1D{
			filters: filters,
			kernelSize: kernelSize,
			strides: 1,
			padding: "valid",
			dataFormat: nil,
			dilationRate: 1,
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
			kernelInitializer: &initializer.GlorotUniform{},
			kernelConstraint: &constraint.NilConstraint{},
			groups: 1,
			kernelRegularizer: &regularizer.NilRegularizer{},
			trainable: true,
			inputs: inputs,
			name: uniqueName("separableconv1d"),		
		}
		for _, option := range options {
			option(s)
		}
		return s
	}
}

type SeparableConv1DOption func (*SeparableConv1D)

func SeparableConv1DWithName(name string) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.name = name
	}
}

func SeparableConv1DWithDtype(dtype DataType) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.dtype = dtype
	}
}

func SeparableConv1DWithTrainable(trainable bool) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.trainable = trainable
	}
}

func SeparableConv1DWithStrides(strides float64) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.strides = strides
	}
}

func SeparableConv1DWithPadding(padding string) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.padding = padding
	}
}

func SeparableConv1DWithDataFormat(dataFormat interface{}) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.dataFormat = dataFormat
	}
}

func SeparableConv1DWithDilationRate(dilationRate float64) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.dilationRate = dilationRate
	}
}

func SeparableConv1DWithDepthMultiplier(depthMultiplier float64) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.depthMultiplier = depthMultiplier
	}
}

func SeparableConv1DWithActivation(activation string) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.activation = activation
	}
}

func SeparableConv1DWithUseBias(useBias bool) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.useBias = useBias
	}
}

func SeparableConv1DWithDepthwiseInitializer(depthwiseInitializer initializer.Initializer) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.depthwiseInitializer = depthwiseInitializer
	}
}

func SeparableConv1DWithPointwiseInitializer(pointwiseInitializer initializer.Initializer) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.pointwiseInitializer = pointwiseInitializer
	}
}

func SeparableConv1DWithBiasInitializer(biasInitializer initializer.Initializer) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.biasInitializer = biasInitializer
	}
}

func SeparableConv1DWithDepthwiseRegularizer(depthwiseRegularizer regularizer.Regularizer) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.depthwiseRegularizer = depthwiseRegularizer
	}
}

func SeparableConv1DWithPointwiseRegularizer(pointwiseRegularizer regularizer.Regularizer) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.pointwiseRegularizer = pointwiseRegularizer
	}
}

func SeparableConv1DWithBiasRegularizer(biasRegularizer regularizer.Regularizer) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.biasRegularizer = biasRegularizer
	}
}

func SeparableConv1DWithActivityRegularizer(activityRegularizer regularizer.Regularizer) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.activityRegularizer = activityRegularizer
	}
}

func SeparableConv1DWithDepthwiseConstraint(depthwiseConstraint constraint.Constraint) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.depthwiseConstraint = depthwiseConstraint
	}
}

func SeparableConv1DWithPointwiseConstraint(pointwiseConstraint constraint.Constraint) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.pointwiseConstraint = pointwiseConstraint
	}
}

func SeparableConv1DWithBiasConstraint(biasConstraint constraint.Constraint) func(s *SeparableConv1D) {
	 return func(s *SeparableConv1D) {
		s.biasConstraint = biasConstraint
	}
}


func (s *SeparableConv1D) GetShape() tf.Shape {
	return s.shape
}

func (s *SeparableConv1D) GetDtype() DataType {
	return s.dtype
}

func (s *SeparableConv1D) SetInput(inputs []Layer) {
	s.inputs = inputs
	s.dtype = inputs[0].GetDtype()
}

func (s *SeparableConv1D) GetInputs() []Layer {
	return s.inputs
}

func (s *SeparableConv1D) GetName() string {
	return s.name
}


type jsonConfigSeparableConv1D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (s *SeparableConv1D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigSeparableConv1D{
		ClassName: "SeparableConv1D",
		Name: s.name,
		Config: map[string]interface{}{
			"depthwise_initializer": s.depthwiseInitializer.GetKerasLayerConfig(),
			"pointwise_initializer": s.pointwiseInitializer.GetKerasLayerConfig(),
			"dtype": s.dtype.String(),
			"kernel_size": s.kernelSize,
			"strides": s.strides,
			"padding": s.padding,
			"data_format": s.dataFormat,
			"bias_constraint": s.biasConstraint.GetKerasLayerConfig(),
			"depthwise_constraint": s.depthwiseConstraint.GetKerasLayerConfig(),
			"pointwise_constraint": s.pointwiseConstraint.GetKerasLayerConfig(),
			"trainable": s.trainable,
			"dilation_rate": s.dilationRate,
			"groups": s.groups,
			"kernel_initializer": s.kernelInitializer.GetKerasLayerConfig(),
			"kernel_constraint": s.kernelConstraint.GetKerasLayerConfig(),
			"pointwise_regularizer": s.pointwiseRegularizer.GetKerasLayerConfig(),
			"name": s.name,
			"filters": s.filters,
			"activation": s.activation,
			"bias_initializer": s.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer": s.biasRegularizer.GetKerasLayerConfig(),
			"use_bias": s.useBias,
			"kernel_regularizer": s.kernelRegularizer.GetKerasLayerConfig(),
			"activity_regularizer": s.activityRegularizer.GetKerasLayerConfig(),
			"depth_multiplier": s.depthMultiplier,
			"depthwise_regularizer": s.depthwiseRegularizer.GetKerasLayerConfig(),
		},
		InboundNodes: inboundNodes,
	}
}
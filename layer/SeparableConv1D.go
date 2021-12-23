package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type SeparableConv1D struct {
	name                 string
	dtype                DataType
	inputs               []Layer
	shape                tf.Shape
	trainable            bool
	filters              float64
	kernelSize           float64
	strides              float64
	padding              string
	dataFormat           interface{}
	dilationRate         float64
	depthMultiplier      float64
	activation           string
	useBias              bool
	depthwiseInitializer initializer.Initializer
	pointwiseInitializer initializer.Initializer
	biasInitializer      initializer.Initializer
	depthwiseRegularizer regularizer.Regularizer
	pointwiseRegularizer regularizer.Regularizer
	biasRegularizer      regularizer.Regularizer
	activityRegularizer  regularizer.Regularizer
	depthwiseConstraint  constraint.Constraint
	pointwiseConstraint  constraint.Constraint
	biasConstraint       constraint.Constraint
	kernelRegularizer    regularizer.Regularizer
	groups               float64
	kernelInitializer    initializer.Initializer
	kernelConstraint     constraint.Constraint
}

func NewSeparableConv1D(filters float64, kernelSize float64, options ...SeparableConv1DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		s := &SeparableConv1D{
			filters:              filters,
			kernelSize:           kernelSize,
			strides:              1,
			padding:              "valid",
			dataFormat:           nil,
			dilationRate:         1,
			depthMultiplier:      1,
			activation:           "linear",
			useBias:              true,
			depthwiseInitializer: &initializer.GlorotUniform{},
			pointwiseInitializer: &initializer.GlorotUniform{},
			biasInitializer:      &initializer.Zeros{},
			depthwiseRegularizer: &regularizer.NilRegularizer{},
			pointwiseRegularizer: &regularizer.NilRegularizer{},
			biasRegularizer:      &regularizer.NilRegularizer{},
			activityRegularizer:  &regularizer.NilRegularizer{},
			depthwiseConstraint:  &constraint.NilConstraint{},
			pointwiseConstraint:  &constraint.NilConstraint{},
			biasConstraint:       &constraint.NilConstraint{},
			kernelRegularizer:    &regularizer.NilRegularizer{},
			groups:               1,
			kernelInitializer:    &initializer.GlorotUniform{},
			kernelConstraint:     &constraint.NilConstraint{},
			trainable:            true,
			inputs:               inputs,
			name:                 UniqueName("separableconv1d"),
		}
		for _, option := range options {
			option(s)
		}
		return s
	}
}

type SeparableConv1DOption func(*SeparableConv1D)

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
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
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
		Name:      s.name,
		Config: map[string]interface{}{
			"activation":            s.activation,
			"activity_regularizer":  s.activityRegularizer.GetKerasLayerConfig(),
			"bias_constraint":       s.biasConstraint.GetKerasLayerConfig(),
			"bias_initializer":      s.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer":      s.biasRegularizer.GetKerasLayerConfig(),
			"data_format":           s.dataFormat,
			"depth_multiplier":      s.depthMultiplier,
			"depthwise_constraint":  s.depthwiseConstraint.GetKerasLayerConfig(),
			"depthwise_initializer": s.depthwiseInitializer.GetKerasLayerConfig(),
			"depthwise_regularizer": s.depthwiseRegularizer.GetKerasLayerConfig(),
			"dilation_rate":         s.dilationRate,
			"dtype":                 s.dtype.String(),
			"filters":               s.filters,
			"groups":                s.groups,
			"kernel_constraint":     s.kernelConstraint.GetKerasLayerConfig(),
			"kernel_initializer":    s.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer":    s.kernelRegularizer.GetKerasLayerConfig(),
			"kernel_size":           s.kernelSize,
			"name":                  s.name,
			"padding":               s.padding,
			"pointwise_constraint":  s.pointwiseConstraint.GetKerasLayerConfig(),
			"pointwise_initializer": s.pointwiseInitializer.GetKerasLayerConfig(),
			"pointwise_regularizer": s.pointwiseRegularizer.GetKerasLayerConfig(),
			"strides":               s.strides,
			"trainable":             s.trainable,
			"use_bias":              s.useBias,
		},
		InboundNodes: inboundNodes,
	}
}

func (s *SeparableConv1D) GetCustomLayerDefinition() string {
	return ``
}

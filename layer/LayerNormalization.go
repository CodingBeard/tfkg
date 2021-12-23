package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type LayerNormalization struct {
	name             string
	dtype            DataType
	inputs           []Layer
	shape            tf.Shape
	trainable        bool
	axis             float64
	epsilon          float64
	center           bool
	scale            bool
	betaInitializer  initializer.Initializer
	gammaInitializer initializer.Initializer
	betaRegularizer  regularizer.Regularizer
	gammaRegularizer regularizer.Regularizer
	betaConstraint   constraint.Constraint
	gammaConstraint  constraint.Constraint
}

func NewLayerNormalization(options ...LayerNormalizationOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		l := &LayerNormalization{
			axis:             -1,
			epsilon:          0.001,
			center:           true,
			scale:            true,
			betaInitializer:  &initializer.Zeros{},
			gammaInitializer: &initializer.Ones{},
			betaRegularizer:  &regularizer.NilRegularizer{},
			gammaRegularizer: &regularizer.NilRegularizer{},
			betaConstraint:   &constraint.NilConstraint{},
			gammaConstraint:  &constraint.NilConstraint{},
			trainable:        true,
			inputs:           inputs,
			name:             UniqueName("layernormalization"),
		}
		for _, option := range options {
			option(l)
		}
		return l
	}
}

type LayerNormalizationOption func(*LayerNormalization)

func LayerNormalizationWithName(name string) func(l *LayerNormalization) {
	return func(l *LayerNormalization) {
		l.name = name
	}
}

func LayerNormalizationWithDtype(dtype DataType) func(l *LayerNormalization) {
	return func(l *LayerNormalization) {
		l.dtype = dtype
	}
}

func LayerNormalizationWithTrainable(trainable bool) func(l *LayerNormalization) {
	return func(l *LayerNormalization) {
		l.trainable = trainable
	}
}

func LayerNormalizationWithAxis(axis float64) func(l *LayerNormalization) {
	return func(l *LayerNormalization) {
		l.axis = axis
	}
}

func LayerNormalizationWithEpsilon(epsilon float64) func(l *LayerNormalization) {
	return func(l *LayerNormalization) {
		l.epsilon = epsilon
	}
}

func LayerNormalizationWithCenter(center bool) func(l *LayerNormalization) {
	return func(l *LayerNormalization) {
		l.center = center
	}
}

func LayerNormalizationWithScale(scale bool) func(l *LayerNormalization) {
	return func(l *LayerNormalization) {
		l.scale = scale
	}
}

func LayerNormalizationWithBetaInitializer(betaInitializer initializer.Initializer) func(l *LayerNormalization) {
	return func(l *LayerNormalization) {
		l.betaInitializer = betaInitializer
	}
}

func LayerNormalizationWithGammaInitializer(gammaInitializer initializer.Initializer) func(l *LayerNormalization) {
	return func(l *LayerNormalization) {
		l.gammaInitializer = gammaInitializer
	}
}

func LayerNormalizationWithBetaRegularizer(betaRegularizer regularizer.Regularizer) func(l *LayerNormalization) {
	return func(l *LayerNormalization) {
		l.betaRegularizer = betaRegularizer
	}
}

func LayerNormalizationWithGammaRegularizer(gammaRegularizer regularizer.Regularizer) func(l *LayerNormalization) {
	return func(l *LayerNormalization) {
		l.gammaRegularizer = gammaRegularizer
	}
}

func LayerNormalizationWithBetaConstraint(betaConstraint constraint.Constraint) func(l *LayerNormalization) {
	return func(l *LayerNormalization) {
		l.betaConstraint = betaConstraint
	}
}

func LayerNormalizationWithGammaConstraint(gammaConstraint constraint.Constraint) func(l *LayerNormalization) {
	return func(l *LayerNormalization) {
		l.gammaConstraint = gammaConstraint
	}
}

func (l *LayerNormalization) GetShape() tf.Shape {
	return l.shape
}

func (l *LayerNormalization) GetDtype() DataType {
	return l.dtype
}

func (l *LayerNormalization) SetInput(inputs []Layer) {
	l.inputs = inputs
	l.dtype = inputs[0].GetDtype()
}

func (l *LayerNormalization) GetInputs() []Layer {
	return l.inputs
}

func (l *LayerNormalization) GetName() string {
	return l.name
}

type jsonConfigLayerNormalization struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LayerNormalization) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range l.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigLayerNormalization{
		ClassName: "LayerNormalization",
		Name:      l.name,
		Config: map[string]interface{}{
			"axis":              l.axis,
			"beta_constraint":   l.betaConstraint.GetKerasLayerConfig(),
			"beta_initializer":  l.betaInitializer.GetKerasLayerConfig(),
			"beta_regularizer":  l.betaRegularizer.GetKerasLayerConfig(),
			"center":            l.center,
			"dtype":             l.dtype.String(),
			"epsilon":           l.epsilon,
			"gamma_constraint":  l.gammaConstraint.GetKerasLayerConfig(),
			"gamma_initializer": l.gammaInitializer.GetKerasLayerConfig(),
			"gamma_regularizer": l.gammaRegularizer.GetKerasLayerConfig(),
			"name":              l.name,
			"scale":             l.scale,
			"trainable":         l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LayerNormalization) GetCustomLayerDefinition() string {
	return ``
}

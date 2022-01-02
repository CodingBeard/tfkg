package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LLayerNormalization struct {
	axis             float64
	betaConstraint   constraint.Constraint
	betaInitializer  initializer.Initializer
	betaRegularizer  regularizer.Regularizer
	center           bool
	dtype            DataType
	epsilon          float64
	gammaConstraint  constraint.Constraint
	gammaInitializer initializer.Initializer
	gammaRegularizer regularizer.Regularizer
	inputs           []Layer
	name             string
	scale            bool
	shape            tf.Shape
	trainable        bool
	layerWeights     []*tf.Tensor
}

func LayerNormalization() *LLayerNormalization {
	return &LLayerNormalization{
		axis:             -1,
		betaConstraint:   &constraint.NilConstraint{},
		betaInitializer:  initializer.Zeros(),
		betaRegularizer:  &regularizer.NilRegularizer{},
		center:           true,
		dtype:            Float32,
		epsilon:          0.001,
		gammaConstraint:  &constraint.NilConstraint{},
		gammaInitializer: initializer.Ones(),
		gammaRegularizer: &regularizer.NilRegularizer{},
		name:             UniqueName("layer_normalization"),
		scale:            true,
		trainable:        true,
	}
}

func (l *LLayerNormalization) SetAxis(axis float64) *LLayerNormalization {
	l.axis = axis
	return l
}

func (l *LLayerNormalization) SetBetaConstraint(betaConstraint constraint.Constraint) *LLayerNormalization {
	l.betaConstraint = betaConstraint
	return l
}

func (l *LLayerNormalization) SetBetaInitializer(betaInitializer initializer.Initializer) *LLayerNormalization {
	l.betaInitializer = betaInitializer
	return l
}

func (l *LLayerNormalization) SetBetaRegularizer(betaRegularizer regularizer.Regularizer) *LLayerNormalization {
	l.betaRegularizer = betaRegularizer
	return l
}

func (l *LLayerNormalization) SetCenter(center bool) *LLayerNormalization {
	l.center = center
	return l
}

func (l *LLayerNormalization) SetDtype(dtype DataType) *LLayerNormalization {
	l.dtype = dtype
	return l
}

func (l *LLayerNormalization) SetEpsilon(epsilon float64) *LLayerNormalization {
	l.epsilon = epsilon
	return l
}

func (l *LLayerNormalization) SetGammaConstraint(gammaConstraint constraint.Constraint) *LLayerNormalization {
	l.gammaConstraint = gammaConstraint
	return l
}

func (l *LLayerNormalization) SetGammaInitializer(gammaInitializer initializer.Initializer) *LLayerNormalization {
	l.gammaInitializer = gammaInitializer
	return l
}

func (l *LLayerNormalization) SetGammaRegularizer(gammaRegularizer regularizer.Regularizer) *LLayerNormalization {
	l.gammaRegularizer = gammaRegularizer
	return l
}

func (l *LLayerNormalization) SetName(name string) *LLayerNormalization {
	l.name = name
	return l
}

func (l *LLayerNormalization) SetScale(scale bool) *LLayerNormalization {
	l.scale = scale
	return l
}

func (l *LLayerNormalization) SetShape(shape tf.Shape) *LLayerNormalization {
	l.shape = shape
	return l
}

func (l *LLayerNormalization) SetTrainable(trainable bool) *LLayerNormalization {
	l.trainable = trainable
	return l
}

func (l *LLayerNormalization) SetLayerWeights(layerWeights []*tf.Tensor) *LLayerNormalization {
	l.layerWeights = layerWeights
	return l
}

func (l *LLayerNormalization) GetShape() tf.Shape {
	return l.shape
}

func (l *LLayerNormalization) GetDtype() DataType {
	return l.dtype
}

func (l *LLayerNormalization) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LLayerNormalization) GetInputs() []Layer {
	return l.inputs
}

func (l *LLayerNormalization) GetName() string {
	return l.name
}

func (l *LLayerNormalization) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLLayerNormalization struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LLayerNormalization) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLLayerNormalization{
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

func (l *LLayerNormalization) GetCustomLayerDefinition() string {
	return ``
}

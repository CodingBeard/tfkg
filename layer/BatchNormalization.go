package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LBatchNormalization struct {
	axis                      float64
	betaConstraint            constraint.Constraint
	betaInitializer           initializer.Initializer
	betaRegularizer           regularizer.Regularizer
	center                    bool
	dtype                     DataType
	epsilon                   float64
	gammaConstraint           constraint.Constraint
	gammaInitializer          initializer.Initializer
	gammaRegularizer          regularizer.Regularizer
	inputs                    []Layer
	momentum                  float64
	movingMeanInitializer     initializer.Initializer
	movingVarianceInitializer initializer.Initializer
	name                      string
	scale                     bool
	shape                     tf.Shape
	trainable                 bool
	layerWeights              []*tf.Tensor
}

func BatchNormalization() *LBatchNormalization {
	return &LBatchNormalization{
		axis:                      -1,
		betaConstraint:            &constraint.NilConstraint{},
		betaInitializer:           initializer.Zeros(),
		betaRegularizer:           &regularizer.NilRegularizer{},
		center:                    true,
		dtype:                     Float32,
		epsilon:                   0.001,
		gammaConstraint:           &constraint.NilConstraint{},
		gammaInitializer:          initializer.Ones(),
		gammaRegularizer:          &regularizer.NilRegularizer{},
		momentum:                  0.99,
		movingMeanInitializer:     initializer.Zeros(),
		movingVarianceInitializer: initializer.Ones(),
		name:                      UniqueName("batch_normalization"),
		scale:                     true,
		trainable:                 true,
	}
}

func (l *LBatchNormalization) SetAxis(axis float64) *LBatchNormalization {
	l.axis = axis
	return l
}

func (l *LBatchNormalization) SetBetaConstraint(betaConstraint constraint.Constraint) *LBatchNormalization {
	l.betaConstraint = betaConstraint
	return l
}

func (l *LBatchNormalization) SetBetaInitializer(betaInitializer initializer.Initializer) *LBatchNormalization {
	l.betaInitializer = betaInitializer
	return l
}

func (l *LBatchNormalization) SetBetaRegularizer(betaRegularizer regularizer.Regularizer) *LBatchNormalization {
	l.betaRegularizer = betaRegularizer
	return l
}

func (l *LBatchNormalization) SetCenter(center bool) *LBatchNormalization {
	l.center = center
	return l
}

func (l *LBatchNormalization) SetDtype(dtype DataType) *LBatchNormalization {
	l.dtype = dtype
	return l
}

func (l *LBatchNormalization) SetEpsilon(epsilon float64) *LBatchNormalization {
	l.epsilon = epsilon
	return l
}

func (l *LBatchNormalization) SetGammaConstraint(gammaConstraint constraint.Constraint) *LBatchNormalization {
	l.gammaConstraint = gammaConstraint
	return l
}

func (l *LBatchNormalization) SetGammaInitializer(gammaInitializer initializer.Initializer) *LBatchNormalization {
	l.gammaInitializer = gammaInitializer
	return l
}

func (l *LBatchNormalization) SetGammaRegularizer(gammaRegularizer regularizer.Regularizer) *LBatchNormalization {
	l.gammaRegularizer = gammaRegularizer
	return l
}

func (l *LBatchNormalization) SetMomentum(momentum float64) *LBatchNormalization {
	l.momentum = momentum
	return l
}

func (l *LBatchNormalization) SetMovingMeanInitializer(movingMeanInitializer initializer.Initializer) *LBatchNormalization {
	l.movingMeanInitializer = movingMeanInitializer
	return l
}

func (l *LBatchNormalization) SetMovingVarianceInitializer(movingVarianceInitializer initializer.Initializer) *LBatchNormalization {
	l.movingVarianceInitializer = movingVarianceInitializer
	return l
}

func (l *LBatchNormalization) SetName(name string) *LBatchNormalization {
	l.name = name
	return l
}

func (l *LBatchNormalization) SetScale(scale bool) *LBatchNormalization {
	l.scale = scale
	return l
}

func (l *LBatchNormalization) SetShape(shape tf.Shape) *LBatchNormalization {
	l.shape = shape
	return l
}

func (l *LBatchNormalization) SetTrainable(trainable bool) *LBatchNormalization {
	l.trainable = trainable
	return l
}

func (l *LBatchNormalization) SetLayerWeights(layerWeights []*tf.Tensor) *LBatchNormalization {
	l.layerWeights = layerWeights
	return l
}

func (l *LBatchNormalization) GetShape() tf.Shape {
	return l.shape
}

func (l *LBatchNormalization) GetDtype() DataType {
	return l.dtype
}

func (l *LBatchNormalization) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LBatchNormalization) GetInputs() []Layer {
	return l.inputs
}

func (l *LBatchNormalization) GetName() string {
	return l.name
}

func (l *LBatchNormalization) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLBatchNormalization struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LBatchNormalization) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLBatchNormalization{
		ClassName: "BatchNormalization",
		Name:      l.name,
		Config: map[string]interface{}{
			"axis":                        l.axis,
			"beta_constraint":             l.betaConstraint.GetKerasLayerConfig(),
			"beta_initializer":            l.betaInitializer.GetKerasLayerConfig(),
			"beta_regularizer":            l.betaRegularizer.GetKerasLayerConfig(),
			"center":                      l.center,
			"dtype":                       l.dtype.String(),
			"epsilon":                     l.epsilon,
			"gamma_constraint":            l.gammaConstraint.GetKerasLayerConfig(),
			"gamma_initializer":           l.gammaInitializer.GetKerasLayerConfig(),
			"gamma_regularizer":           l.gammaRegularizer.GetKerasLayerConfig(),
			"momentum":                    l.momentum,
			"moving_mean_initializer":     l.movingMeanInitializer.GetKerasLayerConfig(),
			"moving_variance_initializer": l.movingVarianceInitializer.GetKerasLayerConfig(),
			"name":                        l.name,
			"scale":                       l.scale,
			"trainable":                   l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LBatchNormalization) GetCustomLayerDefinition() string {
	return ``
}

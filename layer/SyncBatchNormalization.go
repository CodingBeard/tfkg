package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LSyncBatchNormalization struct {
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
}

func SyncBatchNormalization() *LSyncBatchNormalization {
	return &LSyncBatchNormalization{
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
		name:                      UniqueName("sync_batch_normalization"),
		scale:                     true,
		trainable:                 true,
	}
}

func (l *LSyncBatchNormalization) SetAxis(axis float64) *LSyncBatchNormalization {
	l.axis = axis
	return l
}

func (l *LSyncBatchNormalization) SetBetaConstraint(betaConstraint constraint.Constraint) *LSyncBatchNormalization {
	l.betaConstraint = betaConstraint
	return l
}

func (l *LSyncBatchNormalization) SetBetaInitializer(betaInitializer initializer.Initializer) *LSyncBatchNormalization {
	l.betaInitializer = betaInitializer
	return l
}

func (l *LSyncBatchNormalization) SetBetaRegularizer(betaRegularizer regularizer.Regularizer) *LSyncBatchNormalization {
	l.betaRegularizer = betaRegularizer
	return l
}

func (l *LSyncBatchNormalization) SetCenter(center bool) *LSyncBatchNormalization {
	l.center = center
	return l
}

func (l *LSyncBatchNormalization) SetDtype(dtype DataType) *LSyncBatchNormalization {
	l.dtype = dtype
	return l
}

func (l *LSyncBatchNormalization) SetEpsilon(epsilon float64) *LSyncBatchNormalization {
	l.epsilon = epsilon
	return l
}

func (l *LSyncBatchNormalization) SetGammaConstraint(gammaConstraint constraint.Constraint) *LSyncBatchNormalization {
	l.gammaConstraint = gammaConstraint
	return l
}

func (l *LSyncBatchNormalization) SetGammaInitializer(gammaInitializer initializer.Initializer) *LSyncBatchNormalization {
	l.gammaInitializer = gammaInitializer
	return l
}

func (l *LSyncBatchNormalization) SetGammaRegularizer(gammaRegularizer regularizer.Regularizer) *LSyncBatchNormalization {
	l.gammaRegularizer = gammaRegularizer
	return l
}

func (l *LSyncBatchNormalization) SetMomentum(momentum float64) *LSyncBatchNormalization {
	l.momentum = momentum
	return l
}

func (l *LSyncBatchNormalization) SetMovingMeanInitializer(movingMeanInitializer initializer.Initializer) *LSyncBatchNormalization {
	l.movingMeanInitializer = movingMeanInitializer
	return l
}

func (l *LSyncBatchNormalization) SetMovingVarianceInitializer(movingVarianceInitializer initializer.Initializer) *LSyncBatchNormalization {
	l.movingVarianceInitializer = movingVarianceInitializer
	return l
}

func (l *LSyncBatchNormalization) SetName(name string) *LSyncBatchNormalization {
	l.name = name
	return l
}

func (l *LSyncBatchNormalization) SetScale(scale bool) *LSyncBatchNormalization {
	l.scale = scale
	return l
}

func (l *LSyncBatchNormalization) SetShape(shape tf.Shape) *LSyncBatchNormalization {
	l.shape = shape
	return l
}

func (l *LSyncBatchNormalization) SetTrainable(trainable bool) *LSyncBatchNormalization {
	l.trainable = trainable
	return l
}

func (l *LSyncBatchNormalization) GetShape() tf.Shape {
	return l.shape
}

func (l *LSyncBatchNormalization) GetDtype() DataType {
	return l.dtype
}

func (l *LSyncBatchNormalization) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LSyncBatchNormalization) GetInputs() []Layer {
	return l.inputs
}

func (l *LSyncBatchNormalization) GetName() string {
	return l.name
}

type jsonConfigLSyncBatchNormalization struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LSyncBatchNormalization) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLSyncBatchNormalization{
		ClassName: "SyncBatchNormalization",
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

func (l *LSyncBatchNormalization) GetCustomLayerDefinition() string {
	return ``
}

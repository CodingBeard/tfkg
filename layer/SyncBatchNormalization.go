package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type SyncBatchNormalization struct {
	name                      string
	dtype                     DataType
	inputs                    []Layer
	shape                     tf.Shape
	trainable                 bool
	axis                      float64
	momentum                  float64
	epsilon                   float64
	center                    bool
	scale                     bool
	betaInitializer           initializer.Initializer
	gammaInitializer          initializer.Initializer
	movingMeanInitializer     initializer.Initializer
	movingVarianceInitializer initializer.Initializer
	betaRegularizer           regularizer.Regularizer
	gammaRegularizer          regularizer.Regularizer
	betaConstraint            constraint.Constraint
	gammaConstraint           constraint.Constraint
}

func NewSyncBatchNormalization(options ...SyncBatchNormalizationOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		s := &SyncBatchNormalization{
			axis:                      -1,
			momentum:                  0.99,
			epsilon:                   0.001,
			center:                    true,
			scale:                     true,
			betaInitializer:           &initializer.Zeros{},
			gammaInitializer:          &initializer.Ones{},
			movingMeanInitializer:     &initializer.Zeros{},
			movingVarianceInitializer: &initializer.Ones{},
			betaRegularizer:           &regularizer.NilRegularizer{},
			gammaRegularizer:          &regularizer.NilRegularizer{},
			betaConstraint:            &constraint.NilConstraint{},
			gammaConstraint:           &constraint.NilConstraint{},
			trainable:                 true,
			inputs:                    inputs,
			name:                      UniqueName("syncbatchnormalization"),
		}
		for _, option := range options {
			option(s)
		}
		return s
	}
}

type SyncBatchNormalizationOption func(*SyncBatchNormalization)

func SyncBatchNormalizationWithName(name string) func(s *SyncBatchNormalization) {
	return func(s *SyncBatchNormalization) {
		s.name = name
	}
}

func SyncBatchNormalizationWithDtype(dtype DataType) func(s *SyncBatchNormalization) {
	return func(s *SyncBatchNormalization) {
		s.dtype = dtype
	}
}

func SyncBatchNormalizationWithTrainable(trainable bool) func(s *SyncBatchNormalization) {
	return func(s *SyncBatchNormalization) {
		s.trainable = trainable
	}
}

func SyncBatchNormalizationWithAxis(axis float64) func(s *SyncBatchNormalization) {
	return func(s *SyncBatchNormalization) {
		s.axis = axis
	}
}

func SyncBatchNormalizationWithMomentum(momentum float64) func(s *SyncBatchNormalization) {
	return func(s *SyncBatchNormalization) {
		s.momentum = momentum
	}
}

func SyncBatchNormalizationWithEpsilon(epsilon float64) func(s *SyncBatchNormalization) {
	return func(s *SyncBatchNormalization) {
		s.epsilon = epsilon
	}
}

func SyncBatchNormalizationWithCenter(center bool) func(s *SyncBatchNormalization) {
	return func(s *SyncBatchNormalization) {
		s.center = center
	}
}

func SyncBatchNormalizationWithScale(scale bool) func(s *SyncBatchNormalization) {
	return func(s *SyncBatchNormalization) {
		s.scale = scale
	}
}

func SyncBatchNormalizationWithBetaInitializer(betaInitializer initializer.Initializer) func(s *SyncBatchNormalization) {
	return func(s *SyncBatchNormalization) {
		s.betaInitializer = betaInitializer
	}
}

func SyncBatchNormalizationWithGammaInitializer(gammaInitializer initializer.Initializer) func(s *SyncBatchNormalization) {
	return func(s *SyncBatchNormalization) {
		s.gammaInitializer = gammaInitializer
	}
}

func SyncBatchNormalizationWithMovingMeanInitializer(movingMeanInitializer initializer.Initializer) func(s *SyncBatchNormalization) {
	return func(s *SyncBatchNormalization) {
		s.movingMeanInitializer = movingMeanInitializer
	}
}

func SyncBatchNormalizationWithMovingVarianceInitializer(movingVarianceInitializer initializer.Initializer) func(s *SyncBatchNormalization) {
	return func(s *SyncBatchNormalization) {
		s.movingVarianceInitializer = movingVarianceInitializer
	}
}

func SyncBatchNormalizationWithBetaRegularizer(betaRegularizer regularizer.Regularizer) func(s *SyncBatchNormalization) {
	return func(s *SyncBatchNormalization) {
		s.betaRegularizer = betaRegularizer
	}
}

func SyncBatchNormalizationWithGammaRegularizer(gammaRegularizer regularizer.Regularizer) func(s *SyncBatchNormalization) {
	return func(s *SyncBatchNormalization) {
		s.gammaRegularizer = gammaRegularizer
	}
}

func SyncBatchNormalizationWithBetaConstraint(betaConstraint constraint.Constraint) func(s *SyncBatchNormalization) {
	return func(s *SyncBatchNormalization) {
		s.betaConstraint = betaConstraint
	}
}

func SyncBatchNormalizationWithGammaConstraint(gammaConstraint constraint.Constraint) func(s *SyncBatchNormalization) {
	return func(s *SyncBatchNormalization) {
		s.gammaConstraint = gammaConstraint
	}
}

func (s *SyncBatchNormalization) GetShape() tf.Shape {
	return s.shape
}

func (s *SyncBatchNormalization) GetDtype() DataType {
	return s.dtype
}

func (s *SyncBatchNormalization) SetInput(inputs []Layer) {
	s.inputs = inputs
	s.dtype = inputs[0].GetDtype()
}

func (s *SyncBatchNormalization) GetInputs() []Layer {
	return s.inputs
}

func (s *SyncBatchNormalization) GetName() string {
	return s.name
}

type jsonConfigSyncBatchNormalization struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (s *SyncBatchNormalization) GetKerasLayerConfig() interface{} {
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
	return jsonConfigSyncBatchNormalization{
		ClassName: "SyncBatchNormalization",
		Name:      s.name,
		Config: map[string]interface{}{
			"axis":                        s.axis,
			"beta_constraint":             s.betaConstraint.GetKerasLayerConfig(),
			"beta_initializer":            s.betaInitializer.GetKerasLayerConfig(),
			"beta_regularizer":            s.betaRegularizer.GetKerasLayerConfig(),
			"center":                      s.center,
			"dtype":                       s.dtype.String(),
			"epsilon":                     s.epsilon,
			"gamma_constraint":            s.gammaConstraint.GetKerasLayerConfig(),
			"gamma_initializer":           s.gammaInitializer.GetKerasLayerConfig(),
			"gamma_regularizer":           s.gammaRegularizer.GetKerasLayerConfig(),
			"momentum":                    s.momentum,
			"moving_mean_initializer":     s.movingMeanInitializer.GetKerasLayerConfig(),
			"moving_variance_initializer": s.movingVarianceInitializer.GetKerasLayerConfig(),
			"name":                        s.name,
			"scale":                       s.scale,
			"trainable":                   s.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (s *SyncBatchNormalization) GetCustomLayerDefinition() string {
	return ``
}

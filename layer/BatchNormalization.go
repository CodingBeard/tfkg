package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type BatchNormalization struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	axis float64
	momentum float64
	epsilon float64
	center bool
	scale bool
	betaInitializer initializer.Initializer
	gammaInitializer initializer.Initializer
	movingMeanInitializer initializer.Initializer
	movingVarianceInitializer initializer.Initializer
	betaRegularizer regularizer.Regularizer
	gammaRegularizer regularizer.Regularizer
	betaConstraint constraint.Constraint
	gammaConstraint constraint.Constraint
}

func NewBatchNormalization(options ...BatchNormalizationOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		b := &BatchNormalization{
			axis: -1,
			momentum: 0.99,
			epsilon: 0.001,
			center: true,
			scale: true,
			betaInitializer: &initializer.Zeros{},
			gammaInitializer: &initializer.Ones{},
			movingMeanInitializer: &initializer.Zeros{},
			movingVarianceInitializer: &initializer.Ones{},
			betaRegularizer: &regularizer.NilRegularizer{},
			gammaRegularizer: &regularizer.NilRegularizer{},
			betaConstraint: &constraint.NilConstraint{},
			gammaConstraint: &constraint.NilConstraint{},
			trainable: true,
			inputs: inputs,
			name: uniqueName("batchnormalization"),		
		}
		for _, option := range options {
			option(b)
		}
		return b
	}
}

type BatchNormalizationOption func (*BatchNormalization)

func BatchNormalizationWithName(name string) func(b *BatchNormalization) {
	 return func(b *BatchNormalization) {
		b.name = name
	}
}

func BatchNormalizationWithDtype(dtype DataType) func(b *BatchNormalization) {
	 return func(b *BatchNormalization) {
		b.dtype = dtype
	}
}

func BatchNormalizationWithTrainable(trainable bool) func(b *BatchNormalization) {
	 return func(b *BatchNormalization) {
		b.trainable = trainable
	}
}

func BatchNormalizationWithAxis(axis float64) func(b *BatchNormalization) {
	 return func(b *BatchNormalization) {
		b.axis = axis
	}
}

func BatchNormalizationWithMomentum(momentum float64) func(b *BatchNormalization) {
	 return func(b *BatchNormalization) {
		b.momentum = momentum
	}
}

func BatchNormalizationWithEpsilon(epsilon float64) func(b *BatchNormalization) {
	 return func(b *BatchNormalization) {
		b.epsilon = epsilon
	}
}

func BatchNormalizationWithCenter(center bool) func(b *BatchNormalization) {
	 return func(b *BatchNormalization) {
		b.center = center
	}
}

func BatchNormalizationWithScale(scale bool) func(b *BatchNormalization) {
	 return func(b *BatchNormalization) {
		b.scale = scale
	}
}

func BatchNormalizationWithBetaInitializer(betaInitializer initializer.Initializer) func(b *BatchNormalization) {
	 return func(b *BatchNormalization) {
		b.betaInitializer = betaInitializer
	}
}

func BatchNormalizationWithGammaInitializer(gammaInitializer initializer.Initializer) func(b *BatchNormalization) {
	 return func(b *BatchNormalization) {
		b.gammaInitializer = gammaInitializer
	}
}

func BatchNormalizationWithMovingMeanInitializer(movingMeanInitializer initializer.Initializer) func(b *BatchNormalization) {
	 return func(b *BatchNormalization) {
		b.movingMeanInitializer = movingMeanInitializer
	}
}

func BatchNormalizationWithMovingVarianceInitializer(movingVarianceInitializer initializer.Initializer) func(b *BatchNormalization) {
	 return func(b *BatchNormalization) {
		b.movingVarianceInitializer = movingVarianceInitializer
	}
}

func BatchNormalizationWithBetaRegularizer(betaRegularizer regularizer.Regularizer) func(b *BatchNormalization) {
	 return func(b *BatchNormalization) {
		b.betaRegularizer = betaRegularizer
	}
}

func BatchNormalizationWithGammaRegularizer(gammaRegularizer regularizer.Regularizer) func(b *BatchNormalization) {
	 return func(b *BatchNormalization) {
		b.gammaRegularizer = gammaRegularizer
	}
}

func BatchNormalizationWithBetaConstraint(betaConstraint constraint.Constraint) func(b *BatchNormalization) {
	 return func(b *BatchNormalization) {
		b.betaConstraint = betaConstraint
	}
}

func BatchNormalizationWithGammaConstraint(gammaConstraint constraint.Constraint) func(b *BatchNormalization) {
	 return func(b *BatchNormalization) {
		b.gammaConstraint = gammaConstraint
	}
}


func (b *BatchNormalization) GetShape() tf.Shape {
	return b.shape
}

func (b *BatchNormalization) GetDtype() DataType {
	return b.dtype
}

func (b *BatchNormalization) SetInput(inputs []Layer) {
	b.inputs = inputs
	b.dtype = inputs[0].GetDtype()
}

func (b *BatchNormalization) GetInputs() []Layer {
	return b.inputs
}

func (b *BatchNormalization) GetName() string {
	return b.name
}


type jsonConfigBatchNormalization struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (b *BatchNormalization) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range b.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigBatchNormalization{
		ClassName: "BatchNormalization",
		Name: b.name,
		Config: map[string]interface{}{
			"epsilon": b.epsilon,
			"gamma_initializer": b.gammaInitializer.GetKerasLayerConfig(),
			"beta_regularizer": b.betaRegularizer.GetKerasLayerConfig(),
			"gamma_regularizer": b.gammaRegularizer.GetKerasLayerConfig(),
			"beta_constraint": b.betaConstraint.GetKerasLayerConfig(),
			"name": b.name,
			"axis": b.axis,
			"scale": b.scale,
			"beta_initializer": b.betaInitializer.GetKerasLayerConfig(),
			"moving_variance_initializer": b.movingVarianceInitializer.GetKerasLayerConfig(),
			"gamma_constraint": b.gammaConstraint.GetKerasLayerConfig(),
			"dtype": b.dtype.String(),
			"momentum": b.momentum,
			"center": b.center,
			"moving_mean_initializer": b.movingMeanInitializer.GetKerasLayerConfig(),
			"trainable": b.trainable,
		},
		InboundNodes: inboundNodes,
	}
}
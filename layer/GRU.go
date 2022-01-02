package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LGRU struct {
	activation           string
	activityRegularizer  regularizer.Regularizer
	biasConstraint       constraint.Constraint
	biasInitializer      initializer.Initializer
	biasRegularizer      regularizer.Regularizer
	dropout              float64
	dtype                DataType
	goBackwards          bool
	implementation       float64
	inputs               []Layer
	kernelConstraint     constraint.Constraint
	kernelInitializer    initializer.Initializer
	kernelRegularizer    regularizer.Regularizer
	name                 string
	recurrentActivation  string
	recurrentConstraint  constraint.Constraint
	recurrentDropout     float64
	recurrentInitializer initializer.Initializer
	recurrentRegularizer regularizer.Regularizer
	resetAfter           bool
	returnSequences      bool
	returnState          bool
	shape                tf.Shape
	stateful             bool
	timeMajor            bool
	trainable            bool
	units                float64
	unroll               bool
	useBias              bool
	layerWeights         []*tf.Tensor
}

func GRU(units float64) *LGRU {
	return &LGRU{
		activation:           "tanh",
		activityRegularizer:  &regularizer.NilRegularizer{},
		biasConstraint:       &constraint.NilConstraint{},
		biasInitializer:      initializer.Zeros(),
		biasRegularizer:      &regularizer.NilRegularizer{},
		dropout:              0,
		dtype:                Float32,
		goBackwards:          false,
		implementation:       2,
		kernelConstraint:     &constraint.NilConstraint{},
		kernelInitializer:    initializer.GlorotUniform(),
		kernelRegularizer:    &regularizer.NilRegularizer{},
		name:                 UniqueName("gru"),
		recurrentActivation:  "sigmoid",
		recurrentConstraint:  &constraint.NilConstraint{},
		recurrentDropout:     0,
		recurrentInitializer: initializer.Orthogonal(),
		recurrentRegularizer: &regularizer.NilRegularizer{},
		resetAfter:           true,
		returnSequences:      false,
		returnState:          false,
		stateful:             false,
		timeMajor:            false,
		trainable:            true,
		units:                units,
		unroll:               false,
		useBias:              true,
	}
}

func (l *LGRU) SetActivation(activation string) *LGRU {
	l.activation = activation
	return l
}

func (l *LGRU) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LGRU {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LGRU) SetBiasConstraint(biasConstraint constraint.Constraint) *LGRU {
	l.biasConstraint = biasConstraint
	return l
}

func (l *LGRU) SetBiasInitializer(biasInitializer initializer.Initializer) *LGRU {
	l.biasInitializer = biasInitializer
	return l
}

func (l *LGRU) SetBiasRegularizer(biasRegularizer regularizer.Regularizer) *LGRU {
	l.biasRegularizer = biasRegularizer
	return l
}

func (l *LGRU) SetDropout(dropout float64) *LGRU {
	l.dropout = dropout
	return l
}

func (l *LGRU) SetDtype(dtype DataType) *LGRU {
	l.dtype = dtype
	return l
}

func (l *LGRU) SetGoBackwards(goBackwards bool) *LGRU {
	l.goBackwards = goBackwards
	return l
}

func (l *LGRU) SetImplementation(implementation float64) *LGRU {
	l.implementation = implementation
	return l
}

func (l *LGRU) SetKernelConstraint(kernelConstraint constraint.Constraint) *LGRU {
	l.kernelConstraint = kernelConstraint
	return l
}

func (l *LGRU) SetKernelInitializer(kernelInitializer initializer.Initializer) *LGRU {
	l.kernelInitializer = kernelInitializer
	return l
}

func (l *LGRU) SetKernelRegularizer(kernelRegularizer regularizer.Regularizer) *LGRU {
	l.kernelRegularizer = kernelRegularizer
	return l
}

func (l *LGRU) SetName(name string) *LGRU {
	l.name = name
	return l
}

func (l *LGRU) SetRecurrentActivation(recurrentActivation string) *LGRU {
	l.recurrentActivation = recurrentActivation
	return l
}

func (l *LGRU) SetRecurrentConstraint(recurrentConstraint constraint.Constraint) *LGRU {
	l.recurrentConstraint = recurrentConstraint
	return l
}

func (l *LGRU) SetRecurrentDropout(recurrentDropout float64) *LGRU {
	l.recurrentDropout = recurrentDropout
	return l
}

func (l *LGRU) SetRecurrentInitializer(recurrentInitializer initializer.Initializer) *LGRU {
	l.recurrentInitializer = recurrentInitializer
	return l
}

func (l *LGRU) SetRecurrentRegularizer(recurrentRegularizer regularizer.Regularizer) *LGRU {
	l.recurrentRegularizer = recurrentRegularizer
	return l
}

func (l *LGRU) SetResetAfter(resetAfter bool) *LGRU {
	l.resetAfter = resetAfter
	return l
}

func (l *LGRU) SetReturnSequences(returnSequences bool) *LGRU {
	l.returnSequences = returnSequences
	return l
}

func (l *LGRU) SetReturnState(returnState bool) *LGRU {
	l.returnState = returnState
	return l
}

func (l *LGRU) SetShape(shape tf.Shape) *LGRU {
	l.shape = shape
	return l
}

func (l *LGRU) SetStateful(stateful bool) *LGRU {
	l.stateful = stateful
	return l
}

func (l *LGRU) SetTimeMajor(timeMajor bool) *LGRU {
	l.timeMajor = timeMajor
	return l
}

func (l *LGRU) SetTrainable(trainable bool) *LGRU {
	l.trainable = trainable
	return l
}

func (l *LGRU) SetUnroll(unroll bool) *LGRU {
	l.unroll = unroll
	return l
}

func (l *LGRU) SetUseBias(useBias bool) *LGRU {
	l.useBias = useBias
	return l
}

func (l *LGRU) SetLayerWeights(layerWeights []*tf.Tensor) *LGRU {
	l.layerWeights = layerWeights
	return l
}

func (l *LGRU) GetShape() tf.Shape {
	return l.shape
}

func (l *LGRU) GetDtype() DataType {
	return l.dtype
}

func (l *LGRU) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LGRU) GetInputs() []Layer {
	return l.inputs
}

func (l *LGRU) GetName() string {
	return l.name
}

func (l *LGRU) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLGRU struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LGRU) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLGRU{
		ClassName: "GRU",
		Name:      l.name,
		Config: map[string]interface{}{
			"activation":            l.activation,
			"activity_regularizer":  l.activityRegularizer.GetKerasLayerConfig(),
			"bias_constraint":       l.biasConstraint.GetKerasLayerConfig(),
			"bias_initializer":      l.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer":      l.biasRegularizer.GetKerasLayerConfig(),
			"dropout":               l.dropout,
			"dtype":                 l.dtype.String(),
			"go_backwards":          l.goBackwards,
			"implementation":        l.implementation,
			"kernel_constraint":     l.kernelConstraint.GetKerasLayerConfig(),
			"kernel_initializer":    l.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer":    l.kernelRegularizer.GetKerasLayerConfig(),
			"name":                  l.name,
			"recurrent_activation":  l.recurrentActivation,
			"recurrent_constraint":  l.recurrentConstraint.GetKerasLayerConfig(),
			"recurrent_dropout":     l.recurrentDropout,
			"recurrent_initializer": l.recurrentInitializer.GetKerasLayerConfig(),
			"recurrent_regularizer": l.recurrentRegularizer.GetKerasLayerConfig(),
			"reset_after":           l.resetAfter,
			"return_sequences":      l.returnSequences,
			"return_state":          l.returnState,
			"stateful":              l.stateful,
			"time_major":            l.timeMajor,
			"trainable":             l.trainable,
			"units":                 l.units,
			"unroll":                l.unroll,
			"use_bias":              l.useBias,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LGRU) GetCustomLayerDefinition() string {
	return ``
}

package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LSimpleRNN struct {
	activation           string
	activityRegularizer  regularizer.Regularizer
	biasConstraint       constraint.Constraint
	biasInitializer      initializer.Initializer
	biasRegularizer      regularizer.Regularizer
	dropout              float64
	dtype                DataType
	goBackwards          bool
	inputs               []Layer
	kernelConstraint     constraint.Constraint
	kernelInitializer    initializer.Initializer
	kernelRegularizer    regularizer.Regularizer
	name                 string
	recurrentConstraint  constraint.Constraint
	recurrentDropout     float64
	recurrentInitializer initializer.Initializer
	recurrentRegularizer regularizer.Regularizer
	returnSequences      bool
	returnState          bool
	shape                tf.Shape
	stateful             bool
	timeMajor            bool
	trainable            bool
	units                float64
	unroll               bool
	useBias              bool
}

func SimpleRNN(units float64) *LSimpleRNN {
	return &LSimpleRNN{
		activation:           "tanh",
		activityRegularizer:  &regularizer.NilRegularizer{},
		biasConstraint:       &constraint.NilConstraint{},
		biasInitializer:      initializer.Zeros(),
		biasRegularizer:      &regularizer.NilRegularizer{},
		dropout:              0,
		dtype:                Float32,
		goBackwards:          false,
		kernelConstraint:     &constraint.NilConstraint{},
		kernelInitializer:    initializer.GlorotUniform(),
		kernelRegularizer:    &regularizer.NilRegularizer{},
		name:                 UniqueName("simple_rnn"),
		recurrentConstraint:  &constraint.NilConstraint{},
		recurrentDropout:     0,
		recurrentInitializer: initializer.Orthogonal(),
		recurrentRegularizer: &regularizer.NilRegularizer{},
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

func (l *LSimpleRNN) SetActivation(activation string) *LSimpleRNN {
	l.activation = activation
	return l
}

func (l *LSimpleRNN) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LSimpleRNN {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LSimpleRNN) SetBiasConstraint(biasConstraint constraint.Constraint) *LSimpleRNN {
	l.biasConstraint = biasConstraint
	return l
}

func (l *LSimpleRNN) SetBiasInitializer(biasInitializer initializer.Initializer) *LSimpleRNN {
	l.biasInitializer = biasInitializer
	return l
}

func (l *LSimpleRNN) SetBiasRegularizer(biasRegularizer regularizer.Regularizer) *LSimpleRNN {
	l.biasRegularizer = biasRegularizer
	return l
}

func (l *LSimpleRNN) SetDropout(dropout float64) *LSimpleRNN {
	l.dropout = dropout
	return l
}

func (l *LSimpleRNN) SetDtype(dtype DataType) *LSimpleRNN {
	l.dtype = dtype
	return l
}

func (l *LSimpleRNN) SetGoBackwards(goBackwards bool) *LSimpleRNN {
	l.goBackwards = goBackwards
	return l
}

func (l *LSimpleRNN) SetKernelConstraint(kernelConstraint constraint.Constraint) *LSimpleRNN {
	l.kernelConstraint = kernelConstraint
	return l
}

func (l *LSimpleRNN) SetKernelInitializer(kernelInitializer initializer.Initializer) *LSimpleRNN {
	l.kernelInitializer = kernelInitializer
	return l
}

func (l *LSimpleRNN) SetKernelRegularizer(kernelRegularizer regularizer.Regularizer) *LSimpleRNN {
	l.kernelRegularizer = kernelRegularizer
	return l
}

func (l *LSimpleRNN) SetName(name string) *LSimpleRNN {
	l.name = name
	return l
}

func (l *LSimpleRNN) SetRecurrentConstraint(recurrentConstraint constraint.Constraint) *LSimpleRNN {
	l.recurrentConstraint = recurrentConstraint
	return l
}

func (l *LSimpleRNN) SetRecurrentDropout(recurrentDropout float64) *LSimpleRNN {
	l.recurrentDropout = recurrentDropout
	return l
}

func (l *LSimpleRNN) SetRecurrentInitializer(recurrentInitializer initializer.Initializer) *LSimpleRNN {
	l.recurrentInitializer = recurrentInitializer
	return l
}

func (l *LSimpleRNN) SetRecurrentRegularizer(recurrentRegularizer regularizer.Regularizer) *LSimpleRNN {
	l.recurrentRegularizer = recurrentRegularizer
	return l
}

func (l *LSimpleRNN) SetReturnSequences(returnSequences bool) *LSimpleRNN {
	l.returnSequences = returnSequences
	return l
}

func (l *LSimpleRNN) SetReturnState(returnState bool) *LSimpleRNN {
	l.returnState = returnState
	return l
}

func (l *LSimpleRNN) SetShape(shape tf.Shape) *LSimpleRNN {
	l.shape = shape
	return l
}

func (l *LSimpleRNN) SetStateful(stateful bool) *LSimpleRNN {
	l.stateful = stateful
	return l
}

func (l *LSimpleRNN) SetTimeMajor(timeMajor bool) *LSimpleRNN {
	l.timeMajor = timeMajor
	return l
}

func (l *LSimpleRNN) SetTrainable(trainable bool) *LSimpleRNN {
	l.trainable = trainable
	return l
}

func (l *LSimpleRNN) SetUnroll(unroll bool) *LSimpleRNN {
	l.unroll = unroll
	return l
}

func (l *LSimpleRNN) SetUseBias(useBias bool) *LSimpleRNN {
	l.useBias = useBias
	return l
}

func (l *LSimpleRNN) GetShape() tf.Shape {
	return l.shape
}

func (l *LSimpleRNN) GetDtype() DataType {
	return l.dtype
}

func (l *LSimpleRNN) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LSimpleRNN) GetInputs() []Layer {
	return l.inputs
}

func (l *LSimpleRNN) GetName() string {
	return l.name
}

type jsonConfigLSimpleRNN struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LSimpleRNN) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLSimpleRNN{
		ClassName: "SimpleRNN",
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
			"kernel_constraint":     l.kernelConstraint.GetKerasLayerConfig(),
			"kernel_initializer":    l.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer":    l.kernelRegularizer.GetKerasLayerConfig(),
			"name":                  l.name,
			"recurrent_constraint":  l.recurrentConstraint.GetKerasLayerConfig(),
			"recurrent_dropout":     l.recurrentDropout,
			"recurrent_initializer": l.recurrentInitializer.GetKerasLayerConfig(),
			"recurrent_regularizer": l.recurrentRegularizer.GetKerasLayerConfig(),
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

func (l *LSimpleRNN) GetCustomLayerDefinition() string {
	return ``
}

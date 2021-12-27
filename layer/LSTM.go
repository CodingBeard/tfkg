package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LLSTM struct {
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
	returnSequences      bool
	returnState          bool
	shape                tf.Shape
	stateful             bool
	timeMajor            bool
	trainable            bool
	unitForgetBias       bool
	units                float64
	unroll               bool
	useBias              bool
}

func LSTM(units float64) *LLSTM {
	return &LLSTM{
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
		name:                 UniqueName("lstm_1"),
		recurrentActivation:  "sigmoid",
		recurrentConstraint:  &constraint.NilConstraint{},
		recurrentDropout:     0,
		recurrentInitializer: initializer.Orthogonal(),
		recurrentRegularizer: &regularizer.NilRegularizer{},
		returnSequences:      false,
		returnState:          false,
		stateful:             false,
		timeMajor:            false,
		trainable:            true,
		unitForgetBias:       true,
		units:                units,
		unroll:               false,
		useBias:              true,
	}
}

func (l *LLSTM) SetActivation(activation string) *LLSTM {
	l.activation = activation
	return l
}

func (l *LLSTM) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LLSTM {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LLSTM) SetBiasConstraint(biasConstraint constraint.Constraint) *LLSTM {
	l.biasConstraint = biasConstraint
	return l
}

func (l *LLSTM) SetBiasInitializer(biasInitializer initializer.Initializer) *LLSTM {
	l.biasInitializer = biasInitializer
	return l
}

func (l *LLSTM) SetBiasRegularizer(biasRegularizer regularizer.Regularizer) *LLSTM {
	l.biasRegularizer = biasRegularizer
	return l
}

func (l *LLSTM) SetDropout(dropout float64) *LLSTM {
	l.dropout = dropout
	return l
}

func (l *LLSTM) SetDtype(dtype DataType) *LLSTM {
	l.dtype = dtype
	return l
}

func (l *LLSTM) SetGoBackwards(goBackwards bool) *LLSTM {
	l.goBackwards = goBackwards
	return l
}

func (l *LLSTM) SetImplementation(implementation float64) *LLSTM {
	l.implementation = implementation
	return l
}

func (l *LLSTM) SetKernelConstraint(kernelConstraint constraint.Constraint) *LLSTM {
	l.kernelConstraint = kernelConstraint
	return l
}

func (l *LLSTM) SetKernelInitializer(kernelInitializer initializer.Initializer) *LLSTM {
	l.kernelInitializer = kernelInitializer
	return l
}

func (l *LLSTM) SetKernelRegularizer(kernelRegularizer regularizer.Regularizer) *LLSTM {
	l.kernelRegularizer = kernelRegularizer
	return l
}

func (l *LLSTM) SetName(name string) *LLSTM {
	l.name = name
	return l
}

func (l *LLSTM) SetRecurrentActivation(recurrentActivation string) *LLSTM {
	l.recurrentActivation = recurrentActivation
	return l
}

func (l *LLSTM) SetRecurrentConstraint(recurrentConstraint constraint.Constraint) *LLSTM {
	l.recurrentConstraint = recurrentConstraint
	return l
}

func (l *LLSTM) SetRecurrentDropout(recurrentDropout float64) *LLSTM {
	l.recurrentDropout = recurrentDropout
	return l
}

func (l *LLSTM) SetRecurrentInitializer(recurrentInitializer initializer.Initializer) *LLSTM {
	l.recurrentInitializer = recurrentInitializer
	return l
}

func (l *LLSTM) SetRecurrentRegularizer(recurrentRegularizer regularizer.Regularizer) *LLSTM {
	l.recurrentRegularizer = recurrentRegularizer
	return l
}

func (l *LLSTM) SetReturnSequences(returnSequences bool) *LLSTM {
	l.returnSequences = returnSequences
	return l
}

func (l *LLSTM) SetReturnState(returnState bool) *LLSTM {
	l.returnState = returnState
	return l
}

func (l *LLSTM) SetShape(shape tf.Shape) *LLSTM {
	l.shape = shape
	return l
}

func (l *LLSTM) SetStateful(stateful bool) *LLSTM {
	l.stateful = stateful
	return l
}

func (l *LLSTM) SetTimeMajor(timeMajor bool) *LLSTM {
	l.timeMajor = timeMajor
	return l
}

func (l *LLSTM) SetTrainable(trainable bool) *LLSTM {
	l.trainable = trainable
	return l
}

func (l *LLSTM) SetUnitForgetBias(unitForgetBias bool) *LLSTM {
	l.unitForgetBias = unitForgetBias
	return l
}

func (l *LLSTM) SetUnroll(unroll bool) *LLSTM {
	l.unroll = unroll
	return l
}

func (l *LLSTM) SetUseBias(useBias bool) *LLSTM {
	l.useBias = useBias
	return l
}

func (l *LLSTM) GetShape() tf.Shape {
	return l.shape
}

func (l *LLSTM) GetDtype() DataType {
	return l.dtype
}

func (l *LLSTM) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LLSTM) GetInputs() []Layer {
	return l.inputs
}

func (l *LLSTM) GetName() string {
	return l.name
}

type jsonConfigLLSTM struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LLSTM) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLLSTM{
		ClassName: "LSTM",
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
			"return_sequences":      l.returnSequences,
			"return_state":          l.returnState,
			"stateful":              l.stateful,
			"time_major":            l.timeMajor,
			"trainable":             l.trainable,
			"unit_forget_bias":      l.unitForgetBias,
			"units":                 l.units,
			"unroll":                l.unroll,
			"use_bias":              l.useBias,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LLSTM) GetCustomLayerDefinition() string {
	return ``
}

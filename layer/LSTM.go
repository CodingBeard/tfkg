package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type LSTM struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	units float64
	activation string
	recurrentActivation string
	useBias bool
	kernelInitializer initializer.Initializer
	recurrentInitializer initializer.Initializer
	biasInitializer initializer.Initializer
	unitForgetBias bool
	kernelRegularizer regularizer.Regularizer
	recurrentRegularizer regularizer.Regularizer
	biasRegularizer regularizer.Regularizer
	activityRegularizer regularizer.Regularizer
	kernelConstraint constraint.Constraint
	recurrentConstraint constraint.Constraint
	biasConstraint constraint.Constraint
	dropout float64
	recurrentDropout float64
	returnSequences bool
	returnState bool
	goBackwards bool
	stateful bool
	timeMajor bool
	unroll bool
	implementation float64
}

func NewLSTM(units float64, options ...LSTMOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		l := &LSTM{
			units: units,
			activation: "tanh",
			recurrentActivation: "sigmoid",
			useBias: true,
			kernelInitializer: &initializer.GlorotUniform{},
			recurrentInitializer: &initializer.Orthogonal{},
			biasInitializer: &initializer.Zeros{},
			unitForgetBias: true,
			kernelRegularizer: &regularizer.NilRegularizer{},
			recurrentRegularizer: &regularizer.NilRegularizer{},
			biasRegularizer: &regularizer.NilRegularizer{},
			activityRegularizer: &regularizer.NilRegularizer{},
			kernelConstraint: &constraint.NilConstraint{},
			recurrentConstraint: &constraint.NilConstraint{},
			biasConstraint: &constraint.NilConstraint{},
			dropout: 0,
			recurrentDropout: 0,
			returnSequences: false,
			returnState: false,
			goBackwards: false,
			stateful: false,
			timeMajor: false,
			unroll: false,
			implementation: 2,
			trainable: true,
			inputs: inputs,
			name: uniqueName("lstm"),		
		}
		for _, option := range options {
			option(l)
		}
		return l
	}
}

type LSTMOption func (*LSTM)

func LSTMWithName(name string) func(l *LSTM) {
	 return func(l *LSTM) {
		l.name = name
	}
}

func LSTMWithDtype(dtype DataType) func(l *LSTM) {
	 return func(l *LSTM) {
		l.dtype = dtype
	}
}

func LSTMWithTrainable(trainable bool) func(l *LSTM) {
	 return func(l *LSTM) {
		l.trainable = trainable
	}
}

func LSTMWithActivation(activation string) func(l *LSTM) {
	 return func(l *LSTM) {
		l.activation = activation
	}
}

func LSTMWithRecurrentActivation(recurrentActivation string) func(l *LSTM) {
	 return func(l *LSTM) {
		l.recurrentActivation = recurrentActivation
	}
}

func LSTMWithUseBias(useBias bool) func(l *LSTM) {
	 return func(l *LSTM) {
		l.useBias = useBias
	}
}

func LSTMWithKernelInitializer(kernelInitializer initializer.Initializer) func(l *LSTM) {
	 return func(l *LSTM) {
		l.kernelInitializer = kernelInitializer
	}
}

func LSTMWithRecurrentInitializer(recurrentInitializer initializer.Initializer) func(l *LSTM) {
	 return func(l *LSTM) {
		l.recurrentInitializer = recurrentInitializer
	}
}

func LSTMWithBiasInitializer(biasInitializer initializer.Initializer) func(l *LSTM) {
	 return func(l *LSTM) {
		l.biasInitializer = biasInitializer
	}
}

func LSTMWithUnitForgetBias(unitForgetBias bool) func(l *LSTM) {
	 return func(l *LSTM) {
		l.unitForgetBias = unitForgetBias
	}
}

func LSTMWithKernelRegularizer(kernelRegularizer regularizer.Regularizer) func(l *LSTM) {
	 return func(l *LSTM) {
		l.kernelRegularizer = kernelRegularizer
	}
}

func LSTMWithRecurrentRegularizer(recurrentRegularizer regularizer.Regularizer) func(l *LSTM) {
	 return func(l *LSTM) {
		l.recurrentRegularizer = recurrentRegularizer
	}
}

func LSTMWithBiasRegularizer(biasRegularizer regularizer.Regularizer) func(l *LSTM) {
	 return func(l *LSTM) {
		l.biasRegularizer = biasRegularizer
	}
}

func LSTMWithActivityRegularizer(activityRegularizer regularizer.Regularizer) func(l *LSTM) {
	 return func(l *LSTM) {
		l.activityRegularizer = activityRegularizer
	}
}

func LSTMWithKernelConstraint(kernelConstraint constraint.Constraint) func(l *LSTM) {
	 return func(l *LSTM) {
		l.kernelConstraint = kernelConstraint
	}
}

func LSTMWithRecurrentConstraint(recurrentConstraint constraint.Constraint) func(l *LSTM) {
	 return func(l *LSTM) {
		l.recurrentConstraint = recurrentConstraint
	}
}

func LSTMWithBiasConstraint(biasConstraint constraint.Constraint) func(l *LSTM) {
	 return func(l *LSTM) {
		l.biasConstraint = biasConstraint
	}
}

func LSTMWithDropout(dropout float64) func(l *LSTM) {
	 return func(l *LSTM) {
		l.dropout = dropout
	}
}

func LSTMWithRecurrentDropout(recurrentDropout float64) func(l *LSTM) {
	 return func(l *LSTM) {
		l.recurrentDropout = recurrentDropout
	}
}

func LSTMWithReturnSequences(returnSequences bool) func(l *LSTM) {
	 return func(l *LSTM) {
		l.returnSequences = returnSequences
	}
}

func LSTMWithReturnState(returnState bool) func(l *LSTM) {
	 return func(l *LSTM) {
		l.returnState = returnState
	}
}

func LSTMWithGoBackwards(goBackwards bool) func(l *LSTM) {
	 return func(l *LSTM) {
		l.goBackwards = goBackwards
	}
}

func LSTMWithStateful(stateful bool) func(l *LSTM) {
	 return func(l *LSTM) {
		l.stateful = stateful
	}
}

func LSTMWithTimeMajor(timeMajor bool) func(l *LSTM) {
	 return func(l *LSTM) {
		l.timeMajor = timeMajor
	}
}

func LSTMWithUnroll(unroll bool) func(l *LSTM) {
	 return func(l *LSTM) {
		l.unroll = unroll
	}
}


func (l *LSTM) GetShape() tf.Shape {
	return l.shape
}

func (l *LSTM) GetDtype() DataType {
	return l.dtype
}

func (l *LSTM) SetInput(inputs []Layer) {
	l.inputs = inputs
	l.dtype = inputs[0].GetDtype()
}

func (l *LSTM) GetInputs() []Layer {
	return l.inputs
}

func (l *LSTM) GetName() string {
	return l.name
}


type jsonConfigLSTM struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (l *LSTM) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLSTM{
		ClassName: "LSTM",
		Name: l.name,
		Config: map[string]interface{}{
			"kernel_regularizer": l.kernelRegularizer.GetKerasLayerConfig(),
			"dropout": l.dropout,
			"trainable": l.trainable,
			"stateful": l.stateful,
			"unroll": l.unroll,
			"activation": l.activation,
			"recurrent_activation": l.recurrentActivation,
			"name": l.name,
			"bias_regularizer": l.biasRegularizer.GetKerasLayerConfig(),
			"bias_constraint": l.biasConstraint.GetKerasLayerConfig(),
			"time_major": l.timeMajor,
			"recurrent_initializer": l.recurrentInitializer.GetKerasLayerConfig(),
			"bias_initializer": l.biasInitializer.GetKerasLayerConfig(),
			"return_state": l.returnState,
			"go_backwards": l.goBackwards,
			"unit_forget_bias": l.unitForgetBias,
			"recurrent_regularizer": l.recurrentRegularizer.GetKerasLayerConfig(),
			"activity_regularizer": l.activityRegularizer.GetKerasLayerConfig(),
			"dtype": l.dtype.String(),
			"units": l.units,
			"use_bias": l.useBias,
			"return_sequences": l.returnSequences,
			"recurrent_constraint": l.recurrentConstraint.GetKerasLayerConfig(),
			"implementation": l.implementation,
			"recurrent_dropout": l.recurrentDropout,
			"kernel_initializer": l.kernelInitializer.GetKerasLayerConfig(),
			"kernel_constraint": l.kernelConstraint.GetKerasLayerConfig(),
		},
		InboundNodes: inboundNodes,
	}
}
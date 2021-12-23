package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type SimpleRNN struct {
	name                 string
	dtype                DataType
	inputs               []Layer
	shape                tf.Shape
	trainable            bool
	units                float64
	activation           string
	useBias              bool
	kernelInitializer    initializer.Initializer
	recurrentInitializer initializer.Initializer
	biasInitializer      initializer.Initializer
	kernelRegularizer    regularizer.Regularizer
	recurrentRegularizer regularizer.Regularizer
	biasRegularizer      regularizer.Regularizer
	activityRegularizer  regularizer.Regularizer
	kernelConstraint     constraint.Constraint
	recurrentConstraint  constraint.Constraint
	biasConstraint       constraint.Constraint
	dropout              float64
	recurrentDropout     float64
	returnSequences      bool
	returnState          bool
	goBackwards          bool
	stateful             bool
	unroll               bool
	timeMajor            bool
}

func NewSimpleRNN(units float64, options ...SimpleRNNOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		s := &SimpleRNN{
			units:                units,
			activation:           "tanh",
			useBias:              true,
			kernelInitializer:    &initializer.GlorotUniform{},
			recurrentInitializer: &initializer.Orthogonal{},
			biasInitializer:      &initializer.Zeros{},
			kernelRegularizer:    &regularizer.NilRegularizer{},
			recurrentRegularizer: &regularizer.NilRegularizer{},
			biasRegularizer:      &regularizer.NilRegularizer{},
			activityRegularizer:  &regularizer.NilRegularizer{},
			kernelConstraint:     &constraint.NilConstraint{},
			recurrentConstraint:  &constraint.NilConstraint{},
			biasConstraint:       &constraint.NilConstraint{},
			dropout:              0,
			recurrentDropout:     0,
			returnSequences:      false,
			returnState:          false,
			goBackwards:          false,
			stateful:             false,
			unroll:               false,
			timeMajor:            false,
			trainable:            true,
			inputs:               inputs,
			name:                 UniqueName("simplernn"),
		}
		for _, option := range options {
			option(s)
		}
		return s
	}
}

type SimpleRNNOption func(*SimpleRNN)

func SimpleRNNWithName(name string) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.name = name
	}
}

func SimpleRNNWithDtype(dtype DataType) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.dtype = dtype
	}
}

func SimpleRNNWithTrainable(trainable bool) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.trainable = trainable
	}
}

func SimpleRNNWithActivation(activation string) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.activation = activation
	}
}

func SimpleRNNWithUseBias(useBias bool) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.useBias = useBias
	}
}

func SimpleRNNWithKernelInitializer(kernelInitializer initializer.Initializer) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.kernelInitializer = kernelInitializer
	}
}

func SimpleRNNWithRecurrentInitializer(recurrentInitializer initializer.Initializer) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.recurrentInitializer = recurrentInitializer
	}
}

func SimpleRNNWithBiasInitializer(biasInitializer initializer.Initializer) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.biasInitializer = biasInitializer
	}
}

func SimpleRNNWithKernelRegularizer(kernelRegularizer regularizer.Regularizer) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.kernelRegularizer = kernelRegularizer
	}
}

func SimpleRNNWithRecurrentRegularizer(recurrentRegularizer regularizer.Regularizer) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.recurrentRegularizer = recurrentRegularizer
	}
}

func SimpleRNNWithBiasRegularizer(biasRegularizer regularizer.Regularizer) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.biasRegularizer = biasRegularizer
	}
}

func SimpleRNNWithActivityRegularizer(activityRegularizer regularizer.Regularizer) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.activityRegularizer = activityRegularizer
	}
}

func SimpleRNNWithKernelConstraint(kernelConstraint constraint.Constraint) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.kernelConstraint = kernelConstraint
	}
}

func SimpleRNNWithRecurrentConstraint(recurrentConstraint constraint.Constraint) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.recurrentConstraint = recurrentConstraint
	}
}

func SimpleRNNWithBiasConstraint(biasConstraint constraint.Constraint) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.biasConstraint = biasConstraint
	}
}

func SimpleRNNWithDropout(dropout float64) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.dropout = dropout
	}
}

func SimpleRNNWithRecurrentDropout(recurrentDropout float64) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.recurrentDropout = recurrentDropout
	}
}

func SimpleRNNWithReturnSequences(returnSequences bool) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.returnSequences = returnSequences
	}
}

func SimpleRNNWithReturnState(returnState bool) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.returnState = returnState
	}
}

func SimpleRNNWithGoBackwards(goBackwards bool) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.goBackwards = goBackwards
	}
}

func SimpleRNNWithStateful(stateful bool) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.stateful = stateful
	}
}

func SimpleRNNWithUnroll(unroll bool) func(s *SimpleRNN) {
	return func(s *SimpleRNN) {
		s.unroll = unroll
	}
}

func (s *SimpleRNN) GetShape() tf.Shape {
	return s.shape
}

func (s *SimpleRNN) GetDtype() DataType {
	return s.dtype
}

func (s *SimpleRNN) SetInput(inputs []Layer) {
	s.inputs = inputs
	s.dtype = inputs[0].GetDtype()
}

func (s *SimpleRNN) GetInputs() []Layer {
	return s.inputs
}

func (s *SimpleRNN) GetName() string {
	return s.name
}

type jsonConfigSimpleRNN struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (s *SimpleRNN) GetKerasLayerConfig() interface{} {
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
	return jsonConfigSimpleRNN{
		ClassName: "SimpleRNN",
		Name:      s.name,
		Config: map[string]interface{}{
			"activation":            s.activation,
			"activity_regularizer":  s.activityRegularizer.GetKerasLayerConfig(),
			"bias_constraint":       s.biasConstraint.GetKerasLayerConfig(),
			"bias_initializer":      s.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer":      s.biasRegularizer.GetKerasLayerConfig(),
			"dropout":               s.dropout,
			"dtype":                 s.dtype.String(),
			"go_backwards":          s.goBackwards,
			"kernel_constraint":     s.kernelConstraint.GetKerasLayerConfig(),
			"kernel_initializer":    s.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer":    s.kernelRegularizer.GetKerasLayerConfig(),
			"name":                  s.name,
			"recurrent_constraint":  s.recurrentConstraint.GetKerasLayerConfig(),
			"recurrent_dropout":     s.recurrentDropout,
			"recurrent_initializer": s.recurrentInitializer.GetKerasLayerConfig(),
			"recurrent_regularizer": s.recurrentRegularizer.GetKerasLayerConfig(),
			"return_sequences":      s.returnSequences,
			"return_state":          s.returnState,
			"stateful":              s.stateful,
			"time_major":            s.timeMajor,
			"trainable":             s.trainable,
			"units":                 s.units,
			"unroll":                s.unroll,
			"use_bias":              s.useBias,
		},
		InboundNodes: inboundNodes,
	}
}

func (s *SimpleRNN) GetCustomLayerDefinition() string {
	return ``
}

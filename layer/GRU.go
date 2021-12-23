package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type GRU struct {
	name                 string
	dtype                DataType
	inputs               []Layer
	shape                tf.Shape
	trainable            bool
	units                float64
	activation           string
	recurrentActivation  string
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
	resetAfter           bool
	implementation       float64
}

func NewGRU(units float64, options ...GRUOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		g := &GRU{
			units:                units,
			activation:           "tanh",
			recurrentActivation:  "sigmoid",
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
			resetAfter:           true,
			implementation:       2,
			trainable:            true,
			inputs:               inputs,
			name:                 UniqueName("gru"),
		}
		for _, option := range options {
			option(g)
		}
		return g
	}
}

type GRUOption func(*GRU)

func GRUWithName(name string) func(g *GRU) {
	return func(g *GRU) {
		g.name = name
	}
}

func GRUWithDtype(dtype DataType) func(g *GRU) {
	return func(g *GRU) {
		g.dtype = dtype
	}
}

func GRUWithTrainable(trainable bool) func(g *GRU) {
	return func(g *GRU) {
		g.trainable = trainable
	}
}

func GRUWithActivation(activation string) func(g *GRU) {
	return func(g *GRU) {
		g.activation = activation
	}
}

func GRUWithRecurrentActivation(recurrentActivation string) func(g *GRU) {
	return func(g *GRU) {
		g.recurrentActivation = recurrentActivation
	}
}

func GRUWithUseBias(useBias bool) func(g *GRU) {
	return func(g *GRU) {
		g.useBias = useBias
	}
}

func GRUWithKernelInitializer(kernelInitializer initializer.Initializer) func(g *GRU) {
	return func(g *GRU) {
		g.kernelInitializer = kernelInitializer
	}
}

func GRUWithRecurrentInitializer(recurrentInitializer initializer.Initializer) func(g *GRU) {
	return func(g *GRU) {
		g.recurrentInitializer = recurrentInitializer
	}
}

func GRUWithBiasInitializer(biasInitializer initializer.Initializer) func(g *GRU) {
	return func(g *GRU) {
		g.biasInitializer = biasInitializer
	}
}

func GRUWithKernelRegularizer(kernelRegularizer regularizer.Regularizer) func(g *GRU) {
	return func(g *GRU) {
		g.kernelRegularizer = kernelRegularizer
	}
}

func GRUWithRecurrentRegularizer(recurrentRegularizer regularizer.Regularizer) func(g *GRU) {
	return func(g *GRU) {
		g.recurrentRegularizer = recurrentRegularizer
	}
}

func GRUWithBiasRegularizer(biasRegularizer regularizer.Regularizer) func(g *GRU) {
	return func(g *GRU) {
		g.biasRegularizer = biasRegularizer
	}
}

func GRUWithActivityRegularizer(activityRegularizer regularizer.Regularizer) func(g *GRU) {
	return func(g *GRU) {
		g.activityRegularizer = activityRegularizer
	}
}

func GRUWithKernelConstraint(kernelConstraint constraint.Constraint) func(g *GRU) {
	return func(g *GRU) {
		g.kernelConstraint = kernelConstraint
	}
}

func GRUWithRecurrentConstraint(recurrentConstraint constraint.Constraint) func(g *GRU) {
	return func(g *GRU) {
		g.recurrentConstraint = recurrentConstraint
	}
}

func GRUWithBiasConstraint(biasConstraint constraint.Constraint) func(g *GRU) {
	return func(g *GRU) {
		g.biasConstraint = biasConstraint
	}
}

func GRUWithDropout(dropout float64) func(g *GRU) {
	return func(g *GRU) {
		g.dropout = dropout
	}
}

func GRUWithRecurrentDropout(recurrentDropout float64) func(g *GRU) {
	return func(g *GRU) {
		g.recurrentDropout = recurrentDropout
	}
}

func GRUWithReturnSequences(returnSequences bool) func(g *GRU) {
	return func(g *GRU) {
		g.returnSequences = returnSequences
	}
}

func GRUWithReturnState(returnState bool) func(g *GRU) {
	return func(g *GRU) {
		g.returnState = returnState
	}
}

func GRUWithGoBackwards(goBackwards bool) func(g *GRU) {
	return func(g *GRU) {
		g.goBackwards = goBackwards
	}
}

func GRUWithStateful(stateful bool) func(g *GRU) {
	return func(g *GRU) {
		g.stateful = stateful
	}
}

func GRUWithUnroll(unroll bool) func(g *GRU) {
	return func(g *GRU) {
		g.unroll = unroll
	}
}

func GRUWithTimeMajor(timeMajor bool) func(g *GRU) {
	return func(g *GRU) {
		g.timeMajor = timeMajor
	}
}

func GRUWithResetAfter(resetAfter bool) func(g *GRU) {
	return func(g *GRU) {
		g.resetAfter = resetAfter
	}
}

func (g *GRU) GetShape() tf.Shape {
	return g.shape
}

func (g *GRU) GetDtype() DataType {
	return g.dtype
}

func (g *GRU) SetInput(inputs []Layer) {
	g.inputs = inputs
	g.dtype = inputs[0].GetDtype()
}

func (g *GRU) GetInputs() []Layer {
	return g.inputs
}

func (g *GRU) GetName() string {
	return g.name
}

type jsonConfigGRU struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (g *GRU) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range g.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigGRU{
		ClassName: "GRU",
		Name:      g.name,
		Config: map[string]interface{}{
			"activation":            g.activation,
			"activity_regularizer":  g.activityRegularizer.GetKerasLayerConfig(),
			"bias_constraint":       g.biasConstraint.GetKerasLayerConfig(),
			"bias_initializer":      g.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer":      g.biasRegularizer.GetKerasLayerConfig(),
			"dropout":               g.dropout,
			"dtype":                 g.dtype.String(),
			"go_backwards":          g.goBackwards,
			"implementation":        g.implementation,
			"kernel_constraint":     g.kernelConstraint.GetKerasLayerConfig(),
			"kernel_initializer":    g.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer":    g.kernelRegularizer.GetKerasLayerConfig(),
			"name":                  g.name,
			"recurrent_activation":  g.recurrentActivation,
			"recurrent_constraint":  g.recurrentConstraint.GetKerasLayerConfig(),
			"recurrent_dropout":     g.recurrentDropout,
			"recurrent_initializer": g.recurrentInitializer.GetKerasLayerConfig(),
			"recurrent_regularizer": g.recurrentRegularizer.GetKerasLayerConfig(),
			"reset_after":           g.resetAfter,
			"return_sequences":      g.returnSequences,
			"return_state":          g.returnState,
			"stateful":              g.stateful,
			"time_major":            g.timeMajor,
			"trainable":             g.trainable,
			"units":                 g.units,
			"unroll":                g.unroll,
			"use_bias":              g.useBias,
		},
		InboundNodes: inboundNodes,
	}
}

func (g *GRU) GetCustomLayerDefinition() string {
	return ``
}

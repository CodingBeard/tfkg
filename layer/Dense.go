package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LDense struct {
	activation          string
	activityRegularizer regularizer.Regularizer
	biasConstraint      constraint.Constraint
	biasInitializer     initializer.Initializer
	biasRegularizer     regularizer.Regularizer
	dtype               DataType
	inputs              []Layer
	kernelConstraint    constraint.Constraint
	kernelInitializer   initializer.Initializer
	kernelRegularizer   regularizer.Regularizer
	name                string
	shape               tf.Shape
	trainable           bool
	units               float64
	useBias             bool
}

func Dense(units float64) *LDense {
	return &LDense{
		activation:          "linear",
		activityRegularizer: &regularizer.NilRegularizer{},
		biasConstraint:      &constraint.NilConstraint{},
		biasInitializer:     initializer.Zeros(),
		biasRegularizer:     &regularizer.NilRegularizer{},
		dtype:               Float32,
		kernelConstraint:    &constraint.NilConstraint{},
		kernelInitializer:   initializer.GlorotUniform(),
		kernelRegularizer:   &regularizer.NilRegularizer{},
		name:                UniqueName("dense"),
		trainable:           true,
		units:               units,
		useBias:             true,
	}
}

func (l *LDense) SetActivation(activation string) *LDense {
	l.activation = activation
	return l
}

func (l *LDense) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LDense {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LDense) SetBiasConstraint(biasConstraint constraint.Constraint) *LDense {
	l.biasConstraint = biasConstraint
	return l
}

func (l *LDense) SetBiasInitializer(biasInitializer initializer.Initializer) *LDense {
	l.biasInitializer = biasInitializer
	return l
}

func (l *LDense) SetBiasRegularizer(biasRegularizer regularizer.Regularizer) *LDense {
	l.biasRegularizer = biasRegularizer
	return l
}

func (l *LDense) SetDtype(dtype DataType) *LDense {
	l.dtype = dtype
	return l
}

func (l *LDense) SetKernelConstraint(kernelConstraint constraint.Constraint) *LDense {
	l.kernelConstraint = kernelConstraint
	return l
}

func (l *LDense) SetKernelInitializer(kernelInitializer initializer.Initializer) *LDense {
	l.kernelInitializer = kernelInitializer
	return l
}

func (l *LDense) SetKernelRegularizer(kernelRegularizer regularizer.Regularizer) *LDense {
	l.kernelRegularizer = kernelRegularizer
	return l
}

func (l *LDense) SetName(name string) *LDense {
	l.name = name
	return l
}

func (l *LDense) SetShape(shape tf.Shape) *LDense {
	l.shape = shape
	return l
}

func (l *LDense) SetTrainable(trainable bool) *LDense {
	l.trainable = trainable
	return l
}

func (l *LDense) SetUseBias(useBias bool) *LDense {
	l.useBias = useBias
	return l
}

func (l *LDense) GetShape() tf.Shape {
	return l.shape
}

func (l *LDense) GetDtype() DataType {
	return l.dtype
}

func (l *LDense) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LDense) GetInputs() []Layer {
	return l.inputs
}

func (l *LDense) GetName() string {
	return l.name
}

type jsonConfigLDense struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LDense) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLDense{
		ClassName: "Dense",
		Name:      l.name,
		Config: map[string]interface{}{
			"activation":           l.activation,
			"activity_regularizer": l.activityRegularizer.GetKerasLayerConfig(),
			"bias_constraint":      l.biasConstraint.GetKerasLayerConfig(),
			"bias_initializer":     l.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer":     l.biasRegularizer.GetKerasLayerConfig(),
			"dtype":                l.dtype.String(),
			"kernel_constraint":    l.kernelConstraint.GetKerasLayerConfig(),
			"kernel_initializer":   l.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer":   l.kernelRegularizer.GetKerasLayerConfig(),
			"name":                 l.name,
			"trainable":            l.trainable,
			"units":                l.units,
			"use_bias":             l.useBias,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LDense) GetCustomLayerDefinition() string {
	return ``
}

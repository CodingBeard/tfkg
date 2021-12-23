package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type MultiHeadAttention struct {
	name                string
	dtype               DataType
	inputs              []Layer
	shape               tf.Shape
	trainable           bool
	numHeads            float64
	keyDim              float64
	valueDim            interface{}
	dropout             float64
	useBias             bool
	outputShape         interface{}
	attentionAxes       interface{}
	kernelInitializer   initializer.Initializer
	biasInitializer     initializer.Initializer
	kernelRegularizer   regularizer.Regularizer
	biasRegularizer     regularizer.Regularizer
	activityRegularizer regularizer.Regularizer
	kernelConstraint    constraint.Constraint
	biasConstraint      constraint.Constraint
	keyShape            interface{}
	queryShape          interface{}
	valueShape          interface{}
}

func NewMultiHeadAttention(numHeads float64, keyDim float64, options ...MultiHeadAttentionOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		m := &MultiHeadAttention{
			numHeads:            numHeads,
			keyDim:              keyDim,
			valueDim:            nil,
			dropout:             0,
			useBias:             true,
			outputShape:         nil,
			attentionAxes:       nil,
			kernelInitializer:   &initializer.GlorotUniform{},
			biasInitializer:     &initializer.Zeros{},
			kernelRegularizer:   &regularizer.NilRegularizer{},
			biasRegularizer:     &regularizer.NilRegularizer{},
			activityRegularizer: &regularizer.NilRegularizer{},
			kernelConstraint:    &constraint.NilConstraint{},
			biasConstraint:      &constraint.NilConstraint{},
			queryShape:          nil,
			valueShape:          nil,
			keyShape:            nil,
			trainable:           true,
			inputs:              inputs,
			name:                UniqueName("multiheadattention"),
		}
		for _, option := range options {
			option(m)
		}
		return m
	}
}

type MultiHeadAttentionOption func(*MultiHeadAttention)

func MultiHeadAttentionWithName(name string) func(m *MultiHeadAttention) {
	return func(m *MultiHeadAttention) {
		m.name = name
	}
}

func MultiHeadAttentionWithDtype(dtype DataType) func(m *MultiHeadAttention) {
	return func(m *MultiHeadAttention) {
		m.dtype = dtype
	}
}

func MultiHeadAttentionWithTrainable(trainable bool) func(m *MultiHeadAttention) {
	return func(m *MultiHeadAttention) {
		m.trainable = trainable
	}
}

func MultiHeadAttentionWithValueDim(valueDim interface{}) func(m *MultiHeadAttention) {
	return func(m *MultiHeadAttention) {
		m.valueDim = valueDim
	}
}

func MultiHeadAttentionWithDropout(dropout float64) func(m *MultiHeadAttention) {
	return func(m *MultiHeadAttention) {
		m.dropout = dropout
	}
}

func MultiHeadAttentionWithUseBias(useBias bool) func(m *MultiHeadAttention) {
	return func(m *MultiHeadAttention) {
		m.useBias = useBias
	}
}

func MultiHeadAttentionWithOutputShape(outputShape interface{}) func(m *MultiHeadAttention) {
	return func(m *MultiHeadAttention) {
		m.outputShape = outputShape
	}
}

func MultiHeadAttentionWithAttentionAxes(attentionAxes interface{}) func(m *MultiHeadAttention) {
	return func(m *MultiHeadAttention) {
		m.attentionAxes = attentionAxes
	}
}

func MultiHeadAttentionWithKernelInitializer(kernelInitializer initializer.Initializer) func(m *MultiHeadAttention) {
	return func(m *MultiHeadAttention) {
		m.kernelInitializer = kernelInitializer
	}
}

func MultiHeadAttentionWithBiasInitializer(biasInitializer initializer.Initializer) func(m *MultiHeadAttention) {
	return func(m *MultiHeadAttention) {
		m.biasInitializer = biasInitializer
	}
}

func MultiHeadAttentionWithKernelRegularizer(kernelRegularizer regularizer.Regularizer) func(m *MultiHeadAttention) {
	return func(m *MultiHeadAttention) {
		m.kernelRegularizer = kernelRegularizer
	}
}

func MultiHeadAttentionWithBiasRegularizer(biasRegularizer regularizer.Regularizer) func(m *MultiHeadAttention) {
	return func(m *MultiHeadAttention) {
		m.biasRegularizer = biasRegularizer
	}
}

func MultiHeadAttentionWithActivityRegularizer(activityRegularizer regularizer.Regularizer) func(m *MultiHeadAttention) {
	return func(m *MultiHeadAttention) {
		m.activityRegularizer = activityRegularizer
	}
}

func MultiHeadAttentionWithKernelConstraint(kernelConstraint constraint.Constraint) func(m *MultiHeadAttention) {
	return func(m *MultiHeadAttention) {
		m.kernelConstraint = kernelConstraint
	}
}

func MultiHeadAttentionWithBiasConstraint(biasConstraint constraint.Constraint) func(m *MultiHeadAttention) {
	return func(m *MultiHeadAttention) {
		m.biasConstraint = biasConstraint
	}
}

func (m *MultiHeadAttention) GetShape() tf.Shape {
	return m.shape
}

func (m *MultiHeadAttention) GetDtype() DataType {
	return m.dtype
}

func (m *MultiHeadAttention) SetInput(inputs []Layer) {
	m.inputs = inputs
	m.dtype = inputs[0].GetDtype()
}

func (m *MultiHeadAttention) GetInputs() []Layer {
	return m.inputs
}

func (m *MultiHeadAttention) GetName() string {
	return m.name
}

type jsonConfigMultiHeadAttention struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (m *MultiHeadAttention) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range m.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigMultiHeadAttention{
		ClassName: "MultiHeadAttention",
		Name:      m.name,
		Config: map[string]interface{}{
			"activity_regularizer": m.activityRegularizer.GetKerasLayerConfig(),
			"attention_axes":       m.attentionAxes,
			"bias_constraint":      m.biasConstraint.GetKerasLayerConfig(),
			"bias_initializer":     m.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer":     m.biasRegularizer.GetKerasLayerConfig(),
			"dropout":              m.dropout,
			"dtype":                m.dtype.String(),
			"kernel_constraint":    m.kernelConstraint.GetKerasLayerConfig(),
			"kernel_initializer":   m.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer":   m.kernelRegularizer.GetKerasLayerConfig(),
			"key_dim":              m.keyDim,
			"key_shape":            m.keyShape,
			"name":                 m.name,
			"num_heads":            m.numHeads,
			"output_shape":         m.outputShape,
			"query_shape":          m.queryShape,
			"trainable":            m.trainable,
			"use_bias":             m.useBias,
			"value_dim":            m.valueDim,
			"value_shape":          m.valueShape,
		},
		InboundNodes: inboundNodes,
	}
}

func (m *MultiHeadAttention) GetCustomLayerDefinition() string {
	return ``
}

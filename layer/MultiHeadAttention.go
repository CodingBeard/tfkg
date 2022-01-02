package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LMultiHeadAttention struct {
	activityRegularizer regularizer.Regularizer
	attentionAxes       interface{}
	biasConstraint      constraint.Constraint
	biasInitializer     initializer.Initializer
	biasRegularizer     regularizer.Regularizer
	dropout             float64
	dtype               DataType
	inputs              []Layer
	kernelConstraint    constraint.Constraint
	kernelInitializer   initializer.Initializer
	kernelRegularizer   regularizer.Regularizer
	keyDim              float64
	keyShape            interface{}
	name                string
	numHeads            float64
	outputShape         interface{}
	queryShape          interface{}
	shape               tf.Shape
	trainable           bool
	useBias             bool
	valueDim            interface{}
	valueShape          interface{}
	layerWeights        []*tf.Tensor
}

func MultiHeadAttention(keyDim float64, numHeads float64) *LMultiHeadAttention {
	return &LMultiHeadAttention{
		activityRegularizer: &regularizer.NilRegularizer{},
		attentionAxes:       nil,
		biasConstraint:      &constraint.NilConstraint{},
		biasInitializer:     initializer.Zeros(),
		biasRegularizer:     &regularizer.NilRegularizer{},
		dropout:             0,
		dtype:               Float32,
		kernelConstraint:    &constraint.NilConstraint{},
		kernelInitializer:   initializer.GlorotUniform(),
		kernelRegularizer:   &regularizer.NilRegularizer{},
		keyDim:              keyDim,
		keyShape:            nil,
		name:                UniqueName("multi_head_attention"),
		numHeads:            numHeads,
		outputShape:         nil,
		queryShape:          nil,
		trainable:           true,
		useBias:             true,
		valueDim:            nil,
		valueShape:          nil,
	}
}

func (l *LMultiHeadAttention) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LMultiHeadAttention {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LMultiHeadAttention) SetAttentionAxes(attentionAxes interface{}) *LMultiHeadAttention {
	l.attentionAxes = attentionAxes
	return l
}

func (l *LMultiHeadAttention) SetBiasConstraint(biasConstraint constraint.Constraint) *LMultiHeadAttention {
	l.biasConstraint = biasConstraint
	return l
}

func (l *LMultiHeadAttention) SetBiasInitializer(biasInitializer initializer.Initializer) *LMultiHeadAttention {
	l.biasInitializer = biasInitializer
	return l
}

func (l *LMultiHeadAttention) SetBiasRegularizer(biasRegularizer regularizer.Regularizer) *LMultiHeadAttention {
	l.biasRegularizer = biasRegularizer
	return l
}

func (l *LMultiHeadAttention) SetDropout(dropout float64) *LMultiHeadAttention {
	l.dropout = dropout
	return l
}

func (l *LMultiHeadAttention) SetDtype(dtype DataType) *LMultiHeadAttention {
	l.dtype = dtype
	return l
}

func (l *LMultiHeadAttention) SetKernelConstraint(kernelConstraint constraint.Constraint) *LMultiHeadAttention {
	l.kernelConstraint = kernelConstraint
	return l
}

func (l *LMultiHeadAttention) SetKernelInitializer(kernelInitializer initializer.Initializer) *LMultiHeadAttention {
	l.kernelInitializer = kernelInitializer
	return l
}

func (l *LMultiHeadAttention) SetKernelRegularizer(kernelRegularizer regularizer.Regularizer) *LMultiHeadAttention {
	l.kernelRegularizer = kernelRegularizer
	return l
}

func (l *LMultiHeadAttention) SetKeyShape(keyShape interface{}) *LMultiHeadAttention {
	l.keyShape = keyShape
	return l
}

func (l *LMultiHeadAttention) SetName(name string) *LMultiHeadAttention {
	l.name = name
	return l
}

func (l *LMultiHeadAttention) SetOutputShape(outputShape interface{}) *LMultiHeadAttention {
	l.outputShape = outputShape
	return l
}

func (l *LMultiHeadAttention) SetQueryShape(queryShape interface{}) *LMultiHeadAttention {
	l.queryShape = queryShape
	return l
}

func (l *LMultiHeadAttention) SetShape(shape tf.Shape) *LMultiHeadAttention {
	l.shape = shape
	return l
}

func (l *LMultiHeadAttention) SetTrainable(trainable bool) *LMultiHeadAttention {
	l.trainable = trainable
	return l
}

func (l *LMultiHeadAttention) SetUseBias(useBias bool) *LMultiHeadAttention {
	l.useBias = useBias
	return l
}

func (l *LMultiHeadAttention) SetValueDim(valueDim interface{}) *LMultiHeadAttention {
	l.valueDim = valueDim
	return l
}

func (l *LMultiHeadAttention) SetValueShape(valueShape interface{}) *LMultiHeadAttention {
	l.valueShape = valueShape
	return l
}

func (l *LMultiHeadAttention) SetLayerWeights(layerWeights []*tf.Tensor) *LMultiHeadAttention {
	l.layerWeights = layerWeights
	return l
}

func (l *LMultiHeadAttention) GetShape() tf.Shape {
	return l.shape
}

func (l *LMultiHeadAttention) GetDtype() DataType {
	return l.dtype
}

func (l *LMultiHeadAttention) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LMultiHeadAttention) GetInputs() []Layer {
	return l.inputs
}

func (l *LMultiHeadAttention) GetName() string {
	return l.name
}

func (l *LMultiHeadAttention) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLMultiHeadAttention struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LMultiHeadAttention) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLMultiHeadAttention{
		ClassName: "MultiHeadAttention",
		Name:      l.name,
		Config: map[string]interface{}{
			"activity_regularizer": l.activityRegularizer.GetKerasLayerConfig(),
			"attention_axes":       l.attentionAxes,
			"bias_constraint":      l.biasConstraint.GetKerasLayerConfig(),
			"bias_initializer":     l.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer":     l.biasRegularizer.GetKerasLayerConfig(),
			"dropout":              l.dropout,
			"dtype":                l.dtype.String(),
			"kernel_constraint":    l.kernelConstraint.GetKerasLayerConfig(),
			"kernel_initializer":   l.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer":   l.kernelRegularizer.GetKerasLayerConfig(),
			"key_dim":              l.keyDim,
			"key_shape":            l.keyShape,
			"name":                 l.name,
			"num_heads":            l.numHeads,
			"output_shape":         l.outputShape,
			"query_shape":          l.queryShape,
			"trainable":            l.trainable,
			"use_bias":             l.useBias,
			"value_dim":            l.valueDim,
			"value_shape":          l.valueShape,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LMultiHeadAttention) GetCustomLayerDefinition() string {
	return ``
}

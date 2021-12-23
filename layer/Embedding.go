package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type Embedding struct {
	name                  string
	dtype                 DataType
	inputs                []Layer
	shape                 tf.Shape
	trainable             bool
	inputDim              float64
	outputDim             float64
	embeddingsInitializer initializer.Initializer
	embeddingsRegularizer regularizer.Regularizer
	activityRegularizer   regularizer.Regularizer
	embeddingsConstraint  constraint.Constraint
	maskZero              bool
	inputLength           interface{}
	batchInputShape       []interface{}
}

func NewEmbedding(inputDim float64, outputDim float64, options ...EmbeddingOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		e := &Embedding{
			inputDim:              inputDim,
			outputDim:             outputDim,
			embeddingsInitializer: &initializer.RandomUniform{},
			embeddingsRegularizer: &regularizer.NilRegularizer{},
			activityRegularizer:   &regularizer.NilRegularizer{},
			embeddingsConstraint:  &constraint.NilConstraint{},
			maskZero:              false,
			inputLength:           nil,
			batchInputShape:       []interface{}{interface{}(nil), interface{}(nil)},
			trainable:             true,
			inputs:                inputs,
			name:                  UniqueName("embedding"),
		}
		for _, option := range options {
			option(e)
		}
		return e
	}
}

type EmbeddingOption func(*Embedding)

func EmbeddingWithName(name string) func(e *Embedding) {
	return func(e *Embedding) {
		e.name = name
	}
}

func EmbeddingWithDtype(dtype DataType) func(e *Embedding) {
	return func(e *Embedding) {
		e.dtype = dtype
	}
}

func EmbeddingWithTrainable(trainable bool) func(e *Embedding) {
	return func(e *Embedding) {
		e.trainable = trainable
	}
}

func EmbeddingWithEmbeddingsInitializer(embeddingsInitializer initializer.Initializer) func(e *Embedding) {
	return func(e *Embedding) {
		e.embeddingsInitializer = embeddingsInitializer
	}
}

func EmbeddingWithEmbeddingsRegularizer(embeddingsRegularizer regularizer.Regularizer) func(e *Embedding) {
	return func(e *Embedding) {
		e.embeddingsRegularizer = embeddingsRegularizer
	}
}

func EmbeddingWithActivityRegularizer(activityRegularizer regularizer.Regularizer) func(e *Embedding) {
	return func(e *Embedding) {
		e.activityRegularizer = activityRegularizer
	}
}

func EmbeddingWithEmbeddingsConstraint(embeddingsConstraint constraint.Constraint) func(e *Embedding) {
	return func(e *Embedding) {
		e.embeddingsConstraint = embeddingsConstraint
	}
}

func EmbeddingWithMaskZero(maskZero bool) func(e *Embedding) {
	return func(e *Embedding) {
		e.maskZero = maskZero
	}
}

func EmbeddingWithInputLength(inputLength interface{}) func(e *Embedding) {
	return func(e *Embedding) {
		e.inputLength = inputLength
	}
}

func (e *Embedding) GetShape() tf.Shape {
	return e.shape
}

func (e *Embedding) GetDtype() DataType {
	return e.dtype
}

func (e *Embedding) SetInput(inputs []Layer) {
	e.inputs = inputs
	e.dtype = inputs[0].GetDtype()
}

func (e *Embedding) GetInputs() []Layer {
	return e.inputs
}

func (e *Embedding) GetName() string {
	return e.name
}

type jsonConfigEmbedding struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (e *Embedding) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range e.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigEmbedding{
		ClassName: "Embedding",
		Name:      e.name,
		Config: map[string]interface{}{
			"activity_regularizer":   e.activityRegularizer.GetKerasLayerConfig(),
			"batch_input_shape":      e.batchInputShape,
			"dtype":                  e.dtype.String(),
			"embeddings_constraint":  e.embeddingsConstraint.GetKerasLayerConfig(),
			"embeddings_initializer": e.embeddingsInitializer.GetKerasLayerConfig(),
			"embeddings_regularizer": e.embeddingsRegularizer.GetKerasLayerConfig(),
			"input_dim":              e.inputDim,
			"input_length":           e.inputLength,
			"mask_zero":              e.maskZero,
			"name":                   e.name,
			"output_dim":             e.outputDim,
			"trainable":              e.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (e *Embedding) GetCustomLayerDefinition() string {
	return ``
}

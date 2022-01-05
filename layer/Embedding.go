package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LEmbedding struct {
	activityRegularizer   regularizer.Regularizer
	batchInputShape       []interface{}
	dtype                 DataType
	embeddingsConstraint  constraint.Constraint
	embeddingsInitializer string
	embeddingsRegularizer regularizer.Regularizer
	inputDim              float64
	inputLength           interface{}
	inputs                []Layer
	maskZero              bool
	name                  string
	outputDim             float64
	shape                 tf.Shape
	trainable             bool
	layerWeights          []*tf.Tensor
}

func Embedding(inputDim float64, outputDim float64) *LEmbedding {
	return &LEmbedding{
		activityRegularizer:   &regularizer.NilRegularizer{},
		batchInputShape:       []interface{}{interface{}(nil), interface{}(nil)},
		dtype:                 Float32,
		embeddingsConstraint:  &constraint.NilConstraint{},
		embeddingsInitializer: "uniform",
		embeddingsRegularizer: &regularizer.NilRegularizer{},
		inputDim:              inputDim,
		inputLength:           nil,
		maskZero:              false,
		name:                  UniqueName("embedding"),
		outputDim:             outputDim,
		trainable:             true,
	}
}

func (l *LEmbedding) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LEmbedding {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LEmbedding) SetBatchInputShape(batchInputShape []interface{}) *LEmbedding {
	l.batchInputShape = batchInputShape
	return l
}

func (l *LEmbedding) SetDtype(dtype DataType) *LEmbedding {
	l.dtype = dtype
	return l
}

func (l *LEmbedding) SetEmbeddingsConstraint(embeddingsConstraint constraint.Constraint) *LEmbedding {
	l.embeddingsConstraint = embeddingsConstraint
	return l
}

func (l *LEmbedding) SetEmbeddingsInitializer(embeddingsInitializer string) *LEmbedding {
	l.embeddingsInitializer = embeddingsInitializer
	return l
}

func (l *LEmbedding) SetEmbeddingsRegularizer(embeddingsRegularizer regularizer.Regularizer) *LEmbedding {
	l.embeddingsRegularizer = embeddingsRegularizer
	return l
}

func (l *LEmbedding) SetInputLength(inputLength interface{}) *LEmbedding {
	l.inputLength = inputLength
	return l
}

func (l *LEmbedding) SetMaskZero(maskZero bool) *LEmbedding {
	l.maskZero = maskZero
	return l
}

func (l *LEmbedding) SetName(name string) *LEmbedding {
	l.name = name
	return l
}

func (l *LEmbedding) SetShape(shape tf.Shape) *LEmbedding {
	l.shape = shape
	return l
}

func (l *LEmbedding) SetTrainable(trainable bool) *LEmbedding {
	l.trainable = trainable
	return l
}

func (l *LEmbedding) SetLayerWeights(layerWeights []*tf.Tensor) *LEmbedding {
	l.layerWeights = layerWeights
	return l
}

func (l *LEmbedding) GetShape() tf.Shape {
	return l.shape
}

func (l *LEmbedding) GetDtype() DataType {
	return l.dtype
}

func (l *LEmbedding) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LEmbedding) GetInputs() []Layer {
	return l.inputs
}

func (l *LEmbedding) GetName() string {
	return l.name
}

func (l *LEmbedding) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLEmbedding struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LEmbedding) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLEmbedding{
		ClassName: "Embedding",
		Name:      l.name,
		Config: map[string]interface{}{
			"activity_regularizer":   l.activityRegularizer.GetKerasLayerConfig(),
			"batch_input_shape":      l.batchInputShape,
			"dtype":                  l.dtype.String(),
			"embeddings_constraint":  l.embeddingsConstraint.GetKerasLayerConfig(),
			"embeddings_initializer": l.embeddingsInitializer,
			"embeddings_regularizer": l.embeddingsRegularizer.GetKerasLayerConfig(),
			"input_dim":              l.inputDim,
			"input_length":           l.inputLength,
			"mask_zero":              l.maskZero,
			"name":                   l.name,
			"output_dim":             l.outputDim,
			"trainable":              l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LEmbedding) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type TextVectorization struct {
	name                 string
	dtype                DataType
	inputs               []Layer
	shape                tf.Shape
	trainable            bool
	maxTokens            interface{}
	standardize          string
	split                string
	ngrams               interface{}
	outputMode           string
	outputSequenceLength interface{}
	padToMaxTokens       bool
	vocabulary           interface{}
}

func NewTextVectorization(options ...TextVectorizationOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		t := &TextVectorization{
			maxTokens:            nil,
			standardize:          "lower_and_strip_punctuation",
			split:                "whitespace",
			ngrams:               nil,
			outputMode:           "int",
			outputSequenceLength: nil,
			padToMaxTokens:       false,
			vocabulary:           nil,
			trainable:            true,
			inputs:               inputs,
			name:                 UniqueName("textvectorization"),
		}
		for _, option := range options {
			option(t)
		}
		return t
	}
}

type TextVectorizationOption func(*TextVectorization)

func TextVectorizationWithName(name string) func(t *TextVectorization) {
	return func(t *TextVectorization) {
		t.name = name
	}
}

func TextVectorizationWithDtype(dtype DataType) func(t *TextVectorization) {
	return func(t *TextVectorization) {
		t.dtype = dtype
	}
}

func TextVectorizationWithTrainable(trainable bool) func(t *TextVectorization) {
	return func(t *TextVectorization) {
		t.trainable = trainable
	}
}

func TextVectorizationWithMaxTokens(maxTokens interface{}) func(t *TextVectorization) {
	return func(t *TextVectorization) {
		t.maxTokens = maxTokens
	}
}

func TextVectorizationWithStandardize(standardize string) func(t *TextVectorization) {
	return func(t *TextVectorization) {
		t.standardize = standardize
	}
}

func TextVectorizationWithSplit(split string) func(t *TextVectorization) {
	return func(t *TextVectorization) {
		t.split = split
	}
}

func TextVectorizationWithNgrams(ngrams interface{}) func(t *TextVectorization) {
	return func(t *TextVectorization) {
		t.ngrams = ngrams
	}
}

func TextVectorizationWithOutputMode(outputMode string) func(t *TextVectorization) {
	return func(t *TextVectorization) {
		t.outputMode = outputMode
	}
}

func TextVectorizationWithOutputSequenceLength(outputSequenceLength interface{}) func(t *TextVectorization) {
	return func(t *TextVectorization) {
		t.outputSequenceLength = outputSequenceLength
	}
}

func TextVectorizationWithPadToMaxTokens(padToMaxTokens bool) func(t *TextVectorization) {
	return func(t *TextVectorization) {
		t.padToMaxTokens = padToMaxTokens
	}
}

func TextVectorizationWithVocabulary(vocabulary interface{}) func(t *TextVectorization) {
	return func(t *TextVectorization) {
		t.vocabulary = vocabulary
	}
}

func (t *TextVectorization) GetShape() tf.Shape {
	return t.shape
}

func (t *TextVectorization) GetDtype() DataType {
	return t.dtype
}

func (t *TextVectorization) SetInput(inputs []Layer) {
	t.inputs = inputs
	t.dtype = inputs[0].GetDtype()
}

func (t *TextVectorization) GetInputs() []Layer {
	return t.inputs
}

func (t *TextVectorization) GetName() string {
	return t.name
}

type jsonConfigTextVectorization struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (t *TextVectorization) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range t.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigTextVectorization{
		ClassName: "TextVectorization",
		Name:      t.name,
		Config: map[string]interface{}{
			"dtype":                  t.dtype.String(),
			"max_tokens":             t.maxTokens,
			"name":                   t.name,
			"ngrams":                 t.ngrams,
			"output_mode":            t.outputMode,
			"output_sequence_length": t.outputSequenceLength,
			"pad_to_max_tokens":      t.padToMaxTokens,
			"split":                  t.split,
			"standardize":            t.standardize,
			"trainable":              t.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (t *TextVectorization) GetCustomLayerDefinition() string {
	return ``
}

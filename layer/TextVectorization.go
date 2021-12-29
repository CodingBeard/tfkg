package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LTextVectorization struct {
	dtype                DataType
	inputs               []Layer
	maxTokens            interface{}
	name                 string
	ngrams               interface{}
	outputMode           string
	outputSequenceLength interface{}
	padToMaxTokens       bool
	shape                tf.Shape
	split                string
	standardize          string
	trainable            bool
	vocabulary           interface{}
	layerWeights         interface{}
}

func TextVectorization() *LTextVectorization {
	return &LTextVectorization{
		dtype:                String,
		maxTokens:            nil,
		name:                 UniqueName("text_vectorization"),
		ngrams:               nil,
		outputMode:           "int",
		outputSequenceLength: nil,
		padToMaxTokens:       false,
		split:                "whitespace",
		standardize:          "lower_and_strip_punctuation",
		trainable:            true,
		vocabulary:           nil,
	}
}

func (l *LTextVectorization) SetDtype(dtype DataType) *LTextVectorization {
	l.dtype = dtype
	return l
}

func (l *LTextVectorization) SetMaxTokens(maxTokens interface{}) *LTextVectorization {
	l.maxTokens = maxTokens
	return l
}

func (l *LTextVectorization) SetName(name string) *LTextVectorization {
	l.name = name
	return l
}

func (l *LTextVectorization) SetNgrams(ngrams interface{}) *LTextVectorization {
	l.ngrams = ngrams
	return l
}

func (l *LTextVectorization) SetOutputMode(outputMode string) *LTextVectorization {
	l.outputMode = outputMode
	return l
}

func (l *LTextVectorization) SetOutputSequenceLength(outputSequenceLength interface{}) *LTextVectorization {
	l.outputSequenceLength = outputSequenceLength
	return l
}

func (l *LTextVectorization) SetPadToMaxTokens(padToMaxTokens bool) *LTextVectorization {
	l.padToMaxTokens = padToMaxTokens
	return l
}

func (l *LTextVectorization) SetShape(shape tf.Shape) *LTextVectorization {
	l.shape = shape
	return l
}

func (l *LTextVectorization) SetSplit(split string) *LTextVectorization {
	l.split = split
	return l
}

func (l *LTextVectorization) SetStandardize(standardize string) *LTextVectorization {
	l.standardize = standardize
	return l
}

func (l *LTextVectorization) SetTrainable(trainable bool) *LTextVectorization {
	l.trainable = trainable
	return l
}

func (l *LTextVectorization) SetVocabulary(vocabulary interface{}) *LTextVectorization {
	l.vocabulary = vocabulary
	return l
}

func (l *LTextVectorization) SetLayerWeights(layerWeights interface{}) *LTextVectorization {
	l.layerWeights = layerWeights
	return l
}

func (l *LTextVectorization) GetShape() tf.Shape {
	return l.shape
}

func (l *LTextVectorization) GetDtype() DataType {
	return l.dtype
}

func (l *LTextVectorization) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LTextVectorization) GetInputs() []Layer {
	return l.inputs
}

func (l *LTextVectorization) GetName() string {
	return l.name
}

func (l *LTextVectorization) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLTextVectorization struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LTextVectorization) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLTextVectorization{
		ClassName: "TextVectorization",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":                  l.dtype.String(),
			"max_tokens":             l.maxTokens,
			"name":                   l.name,
			"ngrams":                 l.ngrams,
			"output_mode":            l.outputMode,
			"output_sequence_length": l.outputSequenceLength,
			"pad_to_max_tokens":      l.padToMaxTokens,
			"split":                  l.split,
			"standardize":            l.standardize,
			"trainable":              l.trainable,
			"vocabulary":             l.vocabulary,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LTextVectorization) GetCustomLayerDefinition() string {
	return ``
}

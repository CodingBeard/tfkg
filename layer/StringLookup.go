package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LStringLookup struct {
	dtype          DataType
	encoding       interface{}
	inputs         []Layer
	invert         bool
	maskToken      interface{}
	maxTokens      interface{}
	name           string
	numOovIndices  float64
	oovToken       string
	outputMode     string
	padToMaxTokens bool
	shape          tf.Shape
	sparse         bool
	trainable      bool
	vocabulary     interface{}
}

func StringLookup() *LStringLookup {
	return &LStringLookup{
		dtype:          String,
		encoding:       nil,
		invert:         false,
		maskToken:      nil,
		maxTokens:      nil,
		name:           UniqueName("string_lookup"),
		numOovIndices:  1,
		oovToken:       "[UNK]",
		outputMode:     "int",
		padToMaxTokens: false,
		sparse:         false,
		trainable:      true,
		vocabulary:     nil,
	}
}

func (l *LStringLookup) SetDtype(dtype DataType) *LStringLookup {
	l.dtype = dtype
	return l
}

func (l *LStringLookup) SetEncoding(encoding interface{}) *LStringLookup {
	l.encoding = encoding
	return l
}

func (l *LStringLookup) SetInvert(invert bool) *LStringLookup {
	l.invert = invert
	return l
}

func (l *LStringLookup) SetMaskToken(maskToken interface{}) *LStringLookup {
	l.maskToken = maskToken
	return l
}

func (l *LStringLookup) SetMaxTokens(maxTokens interface{}) *LStringLookup {
	l.maxTokens = maxTokens
	return l
}

func (l *LStringLookup) SetName(name string) *LStringLookup {
	l.name = name
	return l
}

func (l *LStringLookup) SetNumOovIndices(numOovIndices float64) *LStringLookup {
	l.numOovIndices = numOovIndices
	return l
}

func (l *LStringLookup) SetOovToken(oovToken string) *LStringLookup {
	l.oovToken = oovToken
	return l
}

func (l *LStringLookup) SetOutputMode(outputMode string) *LStringLookup {
	l.outputMode = outputMode
	return l
}

func (l *LStringLookup) SetPadToMaxTokens(padToMaxTokens bool) *LStringLookup {
	l.padToMaxTokens = padToMaxTokens
	return l
}

func (l *LStringLookup) SetShape(shape tf.Shape) *LStringLookup {
	l.shape = shape
	return l
}

func (l *LStringLookup) SetSparse(sparse bool) *LStringLookup {
	l.sparse = sparse
	return l
}

func (l *LStringLookup) SetTrainable(trainable bool) *LStringLookup {
	l.trainable = trainable
	return l
}

func (l *LStringLookup) SetVocabulary(vocabulary interface{}) *LStringLookup {
	l.vocabulary = vocabulary
	return l
}

func (l *LStringLookup) GetShape() tf.Shape {
	return l.shape
}

func (l *LStringLookup) GetDtype() DataType {
	return l.dtype
}

func (l *LStringLookup) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LStringLookup) GetInputs() []Layer {
	return l.inputs
}

func (l *LStringLookup) GetName() string {
	return l.name
}

type jsonConfigLStringLookup struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LStringLookup) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLStringLookup{
		ClassName: "StringLookup",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":             l.dtype.String(),
			"encoding":          l.encoding,
			"invert":            l.invert,
			"mask_token":        l.maskToken,
			"max_tokens":        l.maxTokens,
			"name":              l.name,
			"num_oov_indices":   l.numOovIndices,
			"oov_token":         l.oovToken,
			"output_mode":       l.outputMode,
			"pad_to_max_tokens": l.padToMaxTokens,
			"sparse":            l.sparse,
			"trainable":         l.trainable,
			"vocabulary":        l.vocabulary,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LStringLookup) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LIntegerLookup struct {
	dtype          DataType
	inputs         []Layer
	invert         bool
	maskToken      interface{}
	maxTokens      interface{}
	name           string
	numOovIndices  float64
	oovToken       float64
	outputMode     string
	padToMaxTokens bool
	shape          tf.Shape
	sparse         bool
	trainable      bool
	vocabulary     interface{}
	layerWeights   []*tf.Tensor
}

func IntegerLookup() *LIntegerLookup {
	return &LIntegerLookup{
		dtype:          Int64,
		invert:         false,
		maskToken:      nil,
		maxTokens:      nil,
		name:           UniqueName("integer_lookup"),
		numOovIndices:  1,
		oovToken:       -1,
		outputMode:     "int",
		padToMaxTokens: false,
		sparse:         false,
		trainable:      true,
		vocabulary:     nil,
	}
}

func (l *LIntegerLookup) SetDtype(dtype DataType) *LIntegerLookup {
	l.dtype = dtype
	return l
}

func (l *LIntegerLookup) SetInvert(invert bool) *LIntegerLookup {
	l.invert = invert
	return l
}

func (l *LIntegerLookup) SetMaskToken(maskToken interface{}) *LIntegerLookup {
	l.maskToken = maskToken
	return l
}

func (l *LIntegerLookup) SetMaxTokens(maxTokens interface{}) *LIntegerLookup {
	l.maxTokens = maxTokens
	return l
}

func (l *LIntegerLookup) SetName(name string) *LIntegerLookup {
	l.name = name
	return l
}

func (l *LIntegerLookup) SetNumOovIndices(numOovIndices float64) *LIntegerLookup {
	l.numOovIndices = numOovIndices
	return l
}

func (l *LIntegerLookup) SetOovToken(oovToken float64) *LIntegerLookup {
	l.oovToken = oovToken
	return l
}

func (l *LIntegerLookup) SetOutputMode(outputMode string) *LIntegerLookup {
	l.outputMode = outputMode
	return l
}

func (l *LIntegerLookup) SetPadToMaxTokens(padToMaxTokens bool) *LIntegerLookup {
	l.padToMaxTokens = padToMaxTokens
	return l
}

func (l *LIntegerLookup) SetShape(shape tf.Shape) *LIntegerLookup {
	l.shape = shape
	return l
}

func (l *LIntegerLookup) SetSparse(sparse bool) *LIntegerLookup {
	l.sparse = sparse
	return l
}

func (l *LIntegerLookup) SetTrainable(trainable bool) *LIntegerLookup {
	l.trainable = trainable
	return l
}

func (l *LIntegerLookup) SetVocabulary(vocabulary interface{}) *LIntegerLookup {
	l.vocabulary = vocabulary
	return l
}

func (l *LIntegerLookup) SetLayerWeights(layerWeights []*tf.Tensor) *LIntegerLookup {
	l.layerWeights = layerWeights
	return l
}

func (l *LIntegerLookup) GetShape() tf.Shape {
	return l.shape
}

func (l *LIntegerLookup) GetDtype() DataType {
	return l.dtype
}

func (l *LIntegerLookup) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LIntegerLookup) GetInputs() []Layer {
	return l.inputs
}

func (l *LIntegerLookup) GetName() string {
	return l.name
}

func (l *LIntegerLookup) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLIntegerLookup struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LIntegerLookup) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLIntegerLookup{
		ClassName: "IntegerLookup",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":             l.dtype.String(),
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

func (l *LIntegerLookup) GetCustomLayerDefinition() string {
	return ``
}

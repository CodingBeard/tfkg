package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type IntegerLookup struct {
	name           string
	dtype          DataType
	inputs         []Layer
	shape          tf.Shape
	trainable      bool
	maxTokens      interface{}
	numOovIndices  float64
	maskToken      interface{}
	oovToken       float64
	vocabulary     interface{}
	invert         bool
	outputMode     string
	sparse         bool
	padToMaxTokens bool
}

func NewIntegerLookup(options ...IntegerLookupOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		i := &IntegerLookup{
			maxTokens:      nil,
			numOovIndices:  1,
			maskToken:      nil,
			oovToken:       -1,
			vocabulary:     nil,
			invert:         false,
			outputMode:     "int",
			sparse:         false,
			padToMaxTokens: false,
			trainable:      true,
			inputs:         inputs,
			name:           UniqueName("integerlookup"),
		}
		for _, option := range options {
			option(i)
		}
		return i
	}
}

type IntegerLookupOption func(*IntegerLookup)

func IntegerLookupWithName(name string) func(i *IntegerLookup) {
	return func(i *IntegerLookup) {
		i.name = name
	}
}

func IntegerLookupWithDtype(dtype DataType) func(i *IntegerLookup) {
	return func(i *IntegerLookup) {
		i.dtype = dtype
	}
}

func IntegerLookupWithTrainable(trainable bool) func(i *IntegerLookup) {
	return func(i *IntegerLookup) {
		i.trainable = trainable
	}
}

func IntegerLookupWithMaxTokens(maxTokens interface{}) func(i *IntegerLookup) {
	return func(i *IntegerLookup) {
		i.maxTokens = maxTokens
	}
}

func IntegerLookupWithNumOovIndices(numOovIndices float64) func(i *IntegerLookup) {
	return func(i *IntegerLookup) {
		i.numOovIndices = numOovIndices
	}
}

func IntegerLookupWithMaskToken(maskToken interface{}) func(i *IntegerLookup) {
	return func(i *IntegerLookup) {
		i.maskToken = maskToken
	}
}

func IntegerLookupWithOovToken(oovToken float64) func(i *IntegerLookup) {
	return func(i *IntegerLookup) {
		i.oovToken = oovToken
	}
}

func IntegerLookupWithVocabulary(vocabulary interface{}) func(i *IntegerLookup) {
	return func(i *IntegerLookup) {
		i.vocabulary = vocabulary
	}
}

func IntegerLookupWithInvert(invert bool) func(i *IntegerLookup) {
	return func(i *IntegerLookup) {
		i.invert = invert
	}
}

func IntegerLookupWithOutputMode(outputMode string) func(i *IntegerLookup) {
	return func(i *IntegerLookup) {
		i.outputMode = outputMode
	}
}

func IntegerLookupWithSparse(sparse bool) func(i *IntegerLookup) {
	return func(i *IntegerLookup) {
		i.sparse = sparse
	}
}

func IntegerLookupWithPadToMaxTokens(padToMaxTokens bool) func(i *IntegerLookup) {
	return func(i *IntegerLookup) {
		i.padToMaxTokens = padToMaxTokens
	}
}

func (i *IntegerLookup) GetShape() tf.Shape {
	return i.shape
}

func (i *IntegerLookup) GetDtype() DataType {
	return i.dtype
}

func (i *IntegerLookup) SetInput(inputs []Layer) {
	i.inputs = inputs
	i.dtype = inputs[0].GetDtype()
}

func (i *IntegerLookup) GetInputs() []Layer {
	return i.inputs
}

func (i *IntegerLookup) GetName() string {
	return i.name
}

type jsonConfigIntegerLookup struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (i *IntegerLookup) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range i.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigIntegerLookup{
		ClassName: "IntegerLookup",
		Name:      i.name,
		Config: map[string]interface{}{
			"dtype":             i.dtype.String(),
			"invert":            i.invert,
			"mask_token":        i.maskToken,
			"max_tokens":        i.maxTokens,
			"name":              i.name,
			"num_oov_indices":   i.numOovIndices,
			"oov_token":         i.oovToken,
			"output_mode":       i.outputMode,
			"pad_to_max_tokens": i.padToMaxTokens,
			"trainable":         i.trainable,
			"vocabulary":        i.vocabulary,
		},
		InboundNodes: inboundNodes,
	}
}

func (i *IntegerLookup) GetCustomLayerDefinition() string {
	return ``
}

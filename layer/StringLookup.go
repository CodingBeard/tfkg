package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type StringLookup struct {
	name           string
	dtype          DataType
	inputs         []Layer
	shape          tf.Shape
	trainable      bool
	maxTokens      interface{}
	numOovIndices  float64
	maskToken      interface{}
	oovToken       string
	vocabulary     interface{}
	encoding       interface{}
	invert         bool
	outputMode     string
	sparse         bool
	padToMaxTokens bool
}

func NewStringLookup(options ...StringLookupOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		s := &StringLookup{
			maxTokens:      nil,
			numOovIndices:  1,
			maskToken:      nil,
			oovToken:       "[UNK]",
			vocabulary:     nil,
			encoding:       nil,
			invert:         false,
			outputMode:     "int",
			sparse:         false,
			padToMaxTokens: false,
			trainable:      true,
			inputs:         inputs,
			name:           UniqueName("stringlookup"),
		}
		for _, option := range options {
			option(s)
		}
		return s
	}
}

type StringLookupOption func(*StringLookup)

func StringLookupWithName(name string) func(s *StringLookup) {
	return func(s *StringLookup) {
		s.name = name
	}
}

func StringLookupWithDtype(dtype DataType) func(s *StringLookup) {
	return func(s *StringLookup) {
		s.dtype = dtype
	}
}

func StringLookupWithTrainable(trainable bool) func(s *StringLookup) {
	return func(s *StringLookup) {
		s.trainable = trainable
	}
}

func StringLookupWithMaxTokens(maxTokens interface{}) func(s *StringLookup) {
	return func(s *StringLookup) {
		s.maxTokens = maxTokens
	}
}

func StringLookupWithNumOovIndices(numOovIndices float64) func(s *StringLookup) {
	return func(s *StringLookup) {
		s.numOovIndices = numOovIndices
	}
}

func StringLookupWithMaskToken(maskToken interface{}) func(s *StringLookup) {
	return func(s *StringLookup) {
		s.maskToken = maskToken
	}
}

func StringLookupWithOovToken(oovToken string) func(s *StringLookup) {
	return func(s *StringLookup) {
		s.oovToken = oovToken
	}
}

func StringLookupWithVocabulary(vocabulary interface{}) func(s *StringLookup) {
	return func(s *StringLookup) {
		s.vocabulary = vocabulary
	}
}

func StringLookupWithEncoding(encoding interface{}) func(s *StringLookup) {
	return func(s *StringLookup) {
		s.encoding = encoding
	}
}

func StringLookupWithInvert(invert bool) func(s *StringLookup) {
	return func(s *StringLookup) {
		s.invert = invert
	}
}

func StringLookupWithOutputMode(outputMode string) func(s *StringLookup) {
	return func(s *StringLookup) {
		s.outputMode = outputMode
	}
}

func StringLookupWithSparse(sparse bool) func(s *StringLookup) {
	return func(s *StringLookup) {
		s.sparse = sparse
	}
}

func StringLookupWithPadToMaxTokens(padToMaxTokens bool) func(s *StringLookup) {
	return func(s *StringLookup) {
		s.padToMaxTokens = padToMaxTokens
	}
}

func (s *StringLookup) GetShape() tf.Shape {
	return s.shape
}

func (s *StringLookup) GetDtype() DataType {
	return s.dtype
}

func (s *StringLookup) SetInput(inputs []Layer) {
	s.inputs = inputs
	s.dtype = inputs[0].GetDtype()
}

func (s *StringLookup) GetInputs() []Layer {
	return s.inputs
}

func (s *StringLookup) GetName() string {
	return s.name
}

type jsonConfigStringLookup struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (s *StringLookup) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range s.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigStringLookup{
		ClassName: "StringLookup",
		Name:      s.name,
		Config: map[string]interface{}{
			"dtype":             s.dtype.String(),
			"encoding":          s.encoding,
			"invert":            s.invert,
			"mask_token":        s.maskToken,
			"max_tokens":        s.maxTokens,
			"name":              s.name,
			"num_oov_indices":   s.numOovIndices,
			"oov_token":         s.oovToken,
			"output_mode":       s.outputMode,
			"pad_to_max_tokens": s.padToMaxTokens,
			"trainable":         s.trainable,
			"vocabulary":        s.vocabulary,
		},
		InboundNodes: inboundNodes,
	}
}

func (s *StringLookup) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LCategoryEncoding struct {
	dtype        DataType
	inputs       []Layer
	name         string
	numTokens    interface{}
	outputMode   string
	shape        tf.Shape
	sparse       bool
	trainable    bool
	layerWeights interface{}
}

func CategoryEncoding() *LCategoryEncoding {
	return &LCategoryEncoding{
		dtype:      Float32,
		name:       UniqueName("category_encoding"),
		numTokens:  nil,
		outputMode: "multi_hot",
		sparse:     false,
		trainable:  true,
	}
}

func (l *LCategoryEncoding) SetDtype(dtype DataType) *LCategoryEncoding {
	l.dtype = dtype
	return l
}

func (l *LCategoryEncoding) SetName(name string) *LCategoryEncoding {
	l.name = name
	return l
}

func (l *LCategoryEncoding) SetNumTokens(numTokens interface{}) *LCategoryEncoding {
	l.numTokens = numTokens
	return l
}

func (l *LCategoryEncoding) SetOutputMode(outputMode string) *LCategoryEncoding {
	l.outputMode = outputMode
	return l
}

func (l *LCategoryEncoding) SetShape(shape tf.Shape) *LCategoryEncoding {
	l.shape = shape
	return l
}

func (l *LCategoryEncoding) SetSparse(sparse bool) *LCategoryEncoding {
	l.sparse = sparse
	return l
}

func (l *LCategoryEncoding) SetTrainable(trainable bool) *LCategoryEncoding {
	l.trainable = trainable
	return l
}

func (l *LCategoryEncoding) SetLayerWeights(layerWeights interface{}) *LCategoryEncoding {
	l.layerWeights = layerWeights
	return l
}

func (l *LCategoryEncoding) GetShape() tf.Shape {
	return l.shape
}

func (l *LCategoryEncoding) GetDtype() DataType {
	return l.dtype
}

func (l *LCategoryEncoding) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LCategoryEncoding) GetInputs() []Layer {
	return l.inputs
}

func (l *LCategoryEncoding) GetName() string {
	return l.name
}

func (l *LCategoryEncoding) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLCategoryEncoding struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LCategoryEncoding) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLCategoryEncoding{
		ClassName: "CategoryEncoding",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":       l.dtype.String(),
			"name":        l.name,
			"num_tokens":  l.numTokens,
			"output_mode": l.outputMode,
			"sparse":      l.sparse,
			"trainable":   l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LCategoryEncoding) GetCustomLayerDefinition() string {
	return ``
}

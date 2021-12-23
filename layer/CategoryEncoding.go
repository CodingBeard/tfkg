package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type CategoryEncoding struct {
	name       string
	dtype      DataType
	inputs     []Layer
	shape      tf.Shape
	trainable  bool
	numTokens  interface{}
	outputMode string
	sparse     bool
}

func NewCategoryEncoding(options ...CategoryEncodingOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		c := &CategoryEncoding{
			numTokens:  nil,
			outputMode: "multi_hot",
			sparse:     false,
			trainable:  true,
			inputs:     inputs,
			name:       UniqueName("categoryencoding"),
		}
		for _, option := range options {
			option(c)
		}
		return c
	}
}

type CategoryEncodingOption func(*CategoryEncoding)

func CategoryEncodingWithName(name string) func(c *CategoryEncoding) {
	return func(c *CategoryEncoding) {
		c.name = name
	}
}

func CategoryEncodingWithDtype(dtype DataType) func(c *CategoryEncoding) {
	return func(c *CategoryEncoding) {
		c.dtype = dtype
	}
}

func CategoryEncodingWithTrainable(trainable bool) func(c *CategoryEncoding) {
	return func(c *CategoryEncoding) {
		c.trainable = trainable
	}
}

func CategoryEncodingWithNumTokens(numTokens interface{}) func(c *CategoryEncoding) {
	return func(c *CategoryEncoding) {
		c.numTokens = numTokens
	}
}

func CategoryEncodingWithOutputMode(outputMode string) func(c *CategoryEncoding) {
	return func(c *CategoryEncoding) {
		c.outputMode = outputMode
	}
}

func CategoryEncodingWithSparse(sparse bool) func(c *CategoryEncoding) {
	return func(c *CategoryEncoding) {
		c.sparse = sparse
	}
}

func (c *CategoryEncoding) GetShape() tf.Shape {
	return c.shape
}

func (c *CategoryEncoding) GetDtype() DataType {
	return c.dtype
}

func (c *CategoryEncoding) SetInput(inputs []Layer) {
	c.inputs = inputs
	c.dtype = inputs[0].GetDtype()
}

func (c *CategoryEncoding) GetInputs() []Layer {
	return c.inputs
}

func (c *CategoryEncoding) GetName() string {
	return c.name
}

type jsonConfigCategoryEncoding struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (c *CategoryEncoding) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range c.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigCategoryEncoding{
		ClassName: "CategoryEncoding",
		Name:      c.name,
		Config: map[string]interface{}{
			"dtype":       c.dtype.String(),
			"name":        c.name,
			"num_tokens":  c.numTokens,
			"output_mode": c.outputMode,
			"sparse":      c.sparse,
			"trainable":   c.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (c *CategoryEncoding) GetCustomLayerDefinition() string {
	return ``
}

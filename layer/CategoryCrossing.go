package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type CategoryCrossing struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
	depth     interface{}
	separator string
}

func NewCategoryCrossing(options ...CategoryCrossingOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		c := &CategoryCrossing{
			depth:     nil,
			separator: "_X_",
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("categorycrossing"),
		}
		for _, option := range options {
			option(c)
		}
		return c
	}
}

type CategoryCrossingOption func(*CategoryCrossing)

func CategoryCrossingWithName(name string) func(c *CategoryCrossing) {
	return func(c *CategoryCrossing) {
		c.name = name
	}
}

func CategoryCrossingWithDtype(dtype DataType) func(c *CategoryCrossing) {
	return func(c *CategoryCrossing) {
		c.dtype = dtype
	}
}

func CategoryCrossingWithTrainable(trainable bool) func(c *CategoryCrossing) {
	return func(c *CategoryCrossing) {
		c.trainable = trainable
	}
}

func CategoryCrossingWithDepth(depth interface{}) func(c *CategoryCrossing) {
	return func(c *CategoryCrossing) {
		c.depth = depth
	}
}

func CategoryCrossingWithSeparator(separator string) func(c *CategoryCrossing) {
	return func(c *CategoryCrossing) {
		c.separator = separator
	}
}

func (c *CategoryCrossing) GetShape() tf.Shape {
	return c.shape
}

func (c *CategoryCrossing) GetDtype() DataType {
	return c.dtype
}

func (c *CategoryCrossing) SetInput(inputs []Layer) {
	c.inputs = inputs
	c.dtype = inputs[0].GetDtype()
}

func (c *CategoryCrossing) GetInputs() []Layer {
	return c.inputs
}

func (c *CategoryCrossing) GetName() string {
	return c.name
}

type jsonConfigCategoryCrossing struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (c *CategoryCrossing) GetKerasLayerConfig() interface{} {
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
	return jsonConfigCategoryCrossing{
		ClassName: "CategoryCrossing",
		Name:      c.name,
		Config: map[string]interface{}{
			"depth":     c.depth,
			"dtype":     c.dtype.String(),
			"name":      c.name,
			"separator": c.separator,
			"trainable": c.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (c *CategoryCrossing) GetCustomLayerDefinition() string {
	return ``
}

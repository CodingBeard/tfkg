package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Cropping2D struct {
	name       string
	dtype      DataType
	inputs     []Layer
	shape      tf.Shape
	trainable  bool
	cropping   []interface{}
	dataFormat interface{}
}

func NewCropping2D(options ...Cropping2DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		c := &Cropping2D{
			cropping:   []interface{}{[]interface{}{0, 0}, []interface{}{0, 0}},
			dataFormat: nil,
			trainable:  true,
			inputs:     inputs,
			name:       UniqueName("cropping2d"),
		}
		for _, option := range options {
			option(c)
		}
		return c
	}
}

type Cropping2DOption func(*Cropping2D)

func Cropping2DWithName(name string) func(c *Cropping2D) {
	return func(c *Cropping2D) {
		c.name = name
	}
}

func Cropping2DWithDtype(dtype DataType) func(c *Cropping2D) {
	return func(c *Cropping2D) {
		c.dtype = dtype
	}
}

func Cropping2DWithTrainable(trainable bool) func(c *Cropping2D) {
	return func(c *Cropping2D) {
		c.trainable = trainable
	}
}

func Cropping2DWithCropping(cropping []interface{}) func(c *Cropping2D) {
	return func(c *Cropping2D) {
		c.cropping = cropping
	}
}

func Cropping2DWithDataFormat(dataFormat interface{}) func(c *Cropping2D) {
	return func(c *Cropping2D) {
		c.dataFormat = dataFormat
	}
}

func (c *Cropping2D) GetShape() tf.Shape {
	return c.shape
}

func (c *Cropping2D) GetDtype() DataType {
	return c.dtype
}

func (c *Cropping2D) SetInput(inputs []Layer) {
	c.inputs = inputs
	c.dtype = inputs[0].GetDtype()
}

func (c *Cropping2D) GetInputs() []Layer {
	return c.inputs
}

func (c *Cropping2D) GetName() string {
	return c.name
}

type jsonConfigCropping2D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (c *Cropping2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigCropping2D{
		ClassName: "Cropping2D",
		Name:      c.name,
		Config: map[string]interface{}{
			"cropping":    c.cropping,
			"data_format": c.dataFormat,
			"dtype":       c.dtype.String(),
			"name":        c.name,
			"trainable":   c.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (c *Cropping2D) GetCustomLayerDefinition() string {
	return ``
}

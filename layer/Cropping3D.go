package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Cropping3D struct {
	name       string
	dtype      DataType
	inputs     []Layer
	shape      tf.Shape
	trainable  bool
	cropping   []interface{}
	dataFormat interface{}
}

func NewCropping3D(options ...Cropping3DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		c := &Cropping3D{
			cropping:   []interface{}{[]interface{}{1, 1}, []interface{}{1, 1}, []interface{}{1, 1}},
			dataFormat: nil,
			trainable:  true,
			inputs:     inputs,
			name:       UniqueName("cropping3d"),
		}
		for _, option := range options {
			option(c)
		}
		return c
	}
}

type Cropping3DOption func(*Cropping3D)

func Cropping3DWithName(name string) func(c *Cropping3D) {
	return func(c *Cropping3D) {
		c.name = name
	}
}

func Cropping3DWithDtype(dtype DataType) func(c *Cropping3D) {
	return func(c *Cropping3D) {
		c.dtype = dtype
	}
}

func Cropping3DWithTrainable(trainable bool) func(c *Cropping3D) {
	return func(c *Cropping3D) {
		c.trainable = trainable
	}
}

func Cropping3DWithCropping(cropping []interface{}) func(c *Cropping3D) {
	return func(c *Cropping3D) {
		c.cropping = cropping
	}
}

func Cropping3DWithDataFormat(dataFormat interface{}) func(c *Cropping3D) {
	return func(c *Cropping3D) {
		c.dataFormat = dataFormat
	}
}

func (c *Cropping3D) GetShape() tf.Shape {
	return c.shape
}

func (c *Cropping3D) GetDtype() DataType {
	return c.dtype
}

func (c *Cropping3D) SetInput(inputs []Layer) {
	c.inputs = inputs
	c.dtype = inputs[0].GetDtype()
}

func (c *Cropping3D) GetInputs() []Layer {
	return c.inputs
}

func (c *Cropping3D) GetName() string {
	return c.name
}

type jsonConfigCropping3D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (c *Cropping3D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigCropping3D{
		ClassName: "Cropping3D",
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

func (c *Cropping3D) GetCustomLayerDefinition() string {
	return ``
}

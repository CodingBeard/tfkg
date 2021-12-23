package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type CenterCrop struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
	height    float64
	width     float64
}

func NewCenterCrop(height float64, width float64, options ...CenterCropOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		c := &CenterCrop{
			height:    height,
			width:     width,
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("centercrop"),
		}
		for _, option := range options {
			option(c)
		}
		return c
	}
}

type CenterCropOption func(*CenterCrop)

func CenterCropWithName(name string) func(c *CenterCrop) {
	return func(c *CenterCrop) {
		c.name = name
	}
}

func CenterCropWithDtype(dtype DataType) func(c *CenterCrop) {
	return func(c *CenterCrop) {
		c.dtype = dtype
	}
}

func CenterCropWithTrainable(trainable bool) func(c *CenterCrop) {
	return func(c *CenterCrop) {
		c.trainable = trainable
	}
}

func (c *CenterCrop) GetShape() tf.Shape {
	return c.shape
}

func (c *CenterCrop) GetDtype() DataType {
	return c.dtype
}

func (c *CenterCrop) SetInput(inputs []Layer) {
	c.inputs = inputs
	c.dtype = inputs[0].GetDtype()
}

func (c *CenterCrop) GetInputs() []Layer {
	return c.inputs
}

func (c *CenterCrop) GetName() string {
	return c.name
}

type jsonConfigCenterCrop struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (c *CenterCrop) GetKerasLayerConfig() interface{} {
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
	return jsonConfigCenterCrop{
		ClassName: "CenterCrop",
		Name:      c.name,
		Config: map[string]interface{}{
			"dtype":     c.dtype.String(),
			"height":    c.height,
			"name":      c.name,
			"trainable": c.trainable,
			"width":     c.width,
		},
		InboundNodes: inboundNodes,
	}
}

func (c *CenterCrop) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Cropping1D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	cropping []interface {}
}

func NewCropping1D(options ...Cropping1DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		c := &Cropping1D{
			cropping: []interface {}{1, 1},
			trainable: true,
			inputs: inputs,
			name: uniqueName("cropping1d"),		
		}
		for _, option := range options {
			option(c)
		}
		return c
	}
}

type Cropping1DOption func (*Cropping1D)

func Cropping1DWithName(name string) func(c *Cropping1D) {
	 return func(c *Cropping1D) {
		c.name = name
	}
}

func Cropping1DWithDtype(dtype DataType) func(c *Cropping1D) {
	 return func(c *Cropping1D) {
		c.dtype = dtype
	}
}

func Cropping1DWithTrainable(trainable bool) func(c *Cropping1D) {
	 return func(c *Cropping1D) {
		c.trainable = trainable
	}
}

func Cropping1DWithCropping(cropping []interface {}) func(c *Cropping1D) {
	 return func(c *Cropping1D) {
		c.cropping = cropping
	}
}


func (c *Cropping1D) GetShape() tf.Shape {
	return c.shape
}

func (c *Cropping1D) GetDtype() DataType {
	return c.dtype
}

func (c *Cropping1D) SetInput(inputs []Layer) {
	c.inputs = inputs
	c.dtype = inputs[0].GetDtype()
}

func (c *Cropping1D) GetInputs() []Layer {
	return c.inputs
}

func (c *Cropping1D) GetName() string {
	return c.name
}


type jsonConfigCropping1D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (c *Cropping1D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigCropping1D{
		ClassName: "Cropping1D",
		Name: c.name,
		Config: map[string]interface{}{
			"name": c.name,
			"trainable": c.trainable,
			"dtype": c.dtype.String(),
			"cropping": c.cropping,
		},
		InboundNodes: inboundNodes,
	}
}
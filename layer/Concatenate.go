package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Concatenate struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	axis float64
}

func NewConcatenate(options ...ConcatenateOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		c := &Concatenate{
			axis: -1,
			trainable: true,
			inputs: inputs,
			name: uniqueName("concatenate"),		
		}
		for _, option := range options {
			option(c)
		}
		return c
	}
}

type ConcatenateOption func (*Concatenate)

func ConcatenateWithName(name string) func(c *Concatenate) {
	 return func(c *Concatenate) {
		c.name = name
	}
}

func ConcatenateWithDtype(dtype DataType) func(c *Concatenate) {
	 return func(c *Concatenate) {
		c.dtype = dtype
	}
}

func ConcatenateWithTrainable(trainable bool) func(c *Concatenate) {
	 return func(c *Concatenate) {
		c.trainable = trainable
	}
}

func ConcatenateWithAxis(axis float64) func(c *Concatenate) {
	 return func(c *Concatenate) {
		c.axis = axis
	}
}


func (c *Concatenate) GetShape() tf.Shape {
	return c.shape
}

func (c *Concatenate) GetDtype() DataType {
	return c.dtype
}

func (c *Concatenate) SetInput(inputs []Layer) {
	c.inputs = inputs
	c.dtype = inputs[0].GetDtype()
}

func (c *Concatenate) GetInputs() []Layer {
	return c.inputs
}

func (c *Concatenate) GetName() string {
	return c.name
}


type jsonConfigConcatenate struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (c *Concatenate) GetKerasLayerConfig() interface{} {
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
	return jsonConfigConcatenate{
		ClassName: "Concatenate",
		Name: c.name,
		Config: map[string]interface{}{
			"name": c.name,
			"trainable": c.trainable,
			"dtype": c.dtype.String(),
			"axis": c.axis,
		},
		InboundNodes: inboundNodes,
	}
}
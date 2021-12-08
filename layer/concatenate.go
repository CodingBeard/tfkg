package layer

import (
	tf "github.com/galeone/tensorflow/tensorflow/go"
)

type Concatenate struct {
	axis   int
	name   string
	dtype  DataType
	inputs []Layer
}

type ConcatenateConfig struct {
	Name string
}

func NewConcatenate(axis int, optionalConfig ...ConcatenateConfig) func(inputs ...Layer) Layer {
	var config ConcatenateConfig
	if len(optionalConfig) == 1 {
		config = optionalConfig[0]
	}

	if config.Name == "" {
		config.Name = uniqueName("concatenate")
	}

	return func(inputs ...Layer) Layer {
		return &Concatenate{
			axis:   axis,
			name:   config.Name,
			inputs: inputs,
		}
	}

}

func (c *Concatenate) GetShape() tf.Shape {
	return tf.MakeShape()
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

type kerasConcatenateConfig struct {
	ClassName string `json:"class_name"`
	Config    struct {
		Name      string `json:"name"`
		Trainable bool   `json:"trainable"`
		Dtype     string `json:"dtype"`
		Axis      int    `json:"axis"`
	} `json:"config"`
	Name         string            `json:"name"`
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
	config := kerasConcatenateConfig{
		ClassName: "ConcatenateLayer",
		Config: struct {
			Name      string `json:"name"`
			Trainable bool   `json:"trainable"`
			Dtype     string `json:"dtype"`
			Axis      int    `json:"axis"`
		}{
			Dtype:     string(c.dtype),
			Name:      c.name,
			Trainable: true,
			Axis:      c.axis,
		},
		Name:         c.name,
		InboundNodes: inboundNodes,
	}

	return config
}

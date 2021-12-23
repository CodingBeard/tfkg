package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Multiply struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
}

func NewMultiply(options ...MultiplyOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		m := &Multiply{
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("multiply"),
		}
		for _, option := range options {
			option(m)
		}
		return m
	}
}

type MultiplyOption func(*Multiply)

func MultiplyWithName(name string) func(m *Multiply) {
	return func(m *Multiply) {
		m.name = name
	}
}

func MultiplyWithDtype(dtype DataType) func(m *Multiply) {
	return func(m *Multiply) {
		m.dtype = dtype
	}
}

func MultiplyWithTrainable(trainable bool) func(m *Multiply) {
	return func(m *Multiply) {
		m.trainable = trainable
	}
}

func (m *Multiply) GetShape() tf.Shape {
	return m.shape
}

func (m *Multiply) GetDtype() DataType {
	return m.dtype
}

func (m *Multiply) SetInput(inputs []Layer) {
	m.inputs = inputs
	m.dtype = inputs[0].GetDtype()
}

func (m *Multiply) GetInputs() []Layer {
	return m.inputs
}

func (m *Multiply) GetName() string {
	return m.name
}

type jsonConfigMultiply struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (m *Multiply) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range m.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigMultiply{
		ClassName: "Multiply",
		Name:      m.name,
		Config: map[string]interface{}{
			"dtype":     m.dtype.String(),
			"name":      m.name,
			"trainable": m.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (m *Multiply) GetCustomLayerDefinition() string {
	return ``
}

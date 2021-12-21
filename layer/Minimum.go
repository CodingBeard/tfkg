package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Minimum struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	
}

func NewMinimum(options ...MinimumOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		m := &Minimum{
			trainable: true,
			inputs: inputs,
			name: uniqueName("minimum"),		
		}
		for _, option := range options {
			option(m)
		}
		return m
	}
}

type MinimumOption func (*Minimum)

func MinimumWithName(name string) func(m *Minimum) {
	 return func(m *Minimum) {
		m.name = name
	}
}

func MinimumWithDtype(dtype DataType) func(m *Minimum) {
	 return func(m *Minimum) {
		m.dtype = dtype
	}
}

func MinimumWithTrainable(trainable bool) func(m *Minimum) {
	 return func(m *Minimum) {
		m.trainable = trainable
	}
}


func (m *Minimum) GetShape() tf.Shape {
	return m.shape
}

func (m *Minimum) GetDtype() DataType {
	return m.dtype
}

func (m *Minimum) SetInput(inputs []Layer) {
	m.inputs = inputs
	m.dtype = inputs[0].GetDtype()
}

func (m *Minimum) GetInputs() []Layer {
	return m.inputs
}

func (m *Minimum) GetName() string {
	return m.name
}


type jsonConfigMinimum struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (m *Minimum) GetKerasLayerConfig() interface{} {
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
	return jsonConfigMinimum{
		ClassName: "Minimum",
		Name: m.name,
		Config: map[string]interface{}{
			"name": m.name,
			"trainable": m.trainable,
			"dtype": m.dtype.String(),
		},
		InboundNodes: inboundNodes,
	}
}
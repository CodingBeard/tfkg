package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Maximum struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	
}

func NewMaximum(options ...MaximumOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		m := &Maximum{
			trainable: true,
			inputs: inputs,
			name: uniqueName("maximum"),		
		}
		for _, option := range options {
			option(m)
		}
		return m
	}
}

type MaximumOption func (*Maximum)

func MaximumWithName(name string) func(m *Maximum) {
	 return func(m *Maximum) {
		m.name = name
	}
}

func MaximumWithDtype(dtype DataType) func(m *Maximum) {
	 return func(m *Maximum) {
		m.dtype = dtype
	}
}

func MaximumWithTrainable(trainable bool) func(m *Maximum) {
	 return func(m *Maximum) {
		m.trainable = trainable
	}
}


func (m *Maximum) GetShape() tf.Shape {
	return m.shape
}

func (m *Maximum) GetDtype() DataType {
	return m.dtype
}

func (m *Maximum) SetInput(inputs []Layer) {
	m.inputs = inputs
	m.dtype = inputs[0].GetDtype()
}

func (m *Maximum) GetInputs() []Layer {
	return m.inputs
}

func (m *Maximum) GetName() string {
	return m.name
}


type jsonConfigMaximum struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (m *Maximum) GetKerasLayerConfig() interface{} {
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
	return jsonConfigMaximum{
		ClassName: "Maximum",
		Name: m.name,
		Config: map[string]interface{}{
			"name": m.name,
			"trainable": m.trainable,
			"dtype": m.dtype.String(),
		},
		InboundNodes: inboundNodes,
	}
}
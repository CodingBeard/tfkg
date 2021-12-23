package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Add struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
}

func NewAdd(options ...AddOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		a := &Add{
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("add"),
		}
		for _, option := range options {
			option(a)
		}
		return a
	}
}

type AddOption func(*Add)

func AddWithName(name string) func(a *Add) {
	return func(a *Add) {
		a.name = name
	}
}

func AddWithDtype(dtype DataType) func(a *Add) {
	return func(a *Add) {
		a.dtype = dtype
	}
}

func AddWithTrainable(trainable bool) func(a *Add) {
	return func(a *Add) {
		a.trainable = trainable
	}
}

func (a *Add) GetShape() tf.Shape {
	return a.shape
}

func (a *Add) GetDtype() DataType {
	return a.dtype
}

func (a *Add) SetInput(inputs []Layer) {
	a.inputs = inputs
	a.dtype = inputs[0].GetDtype()
}

func (a *Add) GetInputs() []Layer {
	return a.inputs
}

func (a *Add) GetName() string {
	return a.name
}

type jsonConfigAdd struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (a *Add) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range a.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigAdd{
		ClassName: "Add",
		Name:      a.name,
		Config: map[string]interface{}{
			"dtype":     a.dtype.String(),
			"name":      a.name,
			"trainable": a.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (a *Add) GetCustomLayerDefinition() string {
	return ``
}

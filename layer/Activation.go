package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Activation struct {
	name       string
	dtype      DataType
	inputs     []Layer
	shape      tf.Shape
	trainable  bool
	activation string
}

func NewActivation(activation string, options ...ActivationOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		a := &Activation{
			activation: activation,
			trainable:  true,
			inputs:     inputs,
			name:       UniqueName("activation"),
		}
		for _, option := range options {
			option(a)
		}
		return a
	}
}

type ActivationOption func(*Activation)

func ActivationWithName(name string) func(a *Activation) {
	return func(a *Activation) {
		a.name = name
	}
}

func ActivationWithDtype(dtype DataType) func(a *Activation) {
	return func(a *Activation) {
		a.dtype = dtype
	}
}

func ActivationWithTrainable(trainable bool) func(a *Activation) {
	return func(a *Activation) {
		a.trainable = trainable
	}
}

func (a *Activation) GetShape() tf.Shape {
	return a.shape
}

func (a *Activation) GetDtype() DataType {
	return a.dtype
}

func (a *Activation) SetInput(inputs []Layer) {
	a.inputs = inputs
	a.dtype = inputs[0].GetDtype()
}

func (a *Activation) GetInputs() []Layer {
	return a.inputs
}

func (a *Activation) GetName() string {
	return a.name
}

type jsonConfigActivation struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (a *Activation) GetKerasLayerConfig() interface{} {
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
	return jsonConfigActivation{
		ClassName: "Activation",
		Name:      a.name,
		Config: map[string]interface{}{
			"activation": a.activation,
			"dtype":      a.dtype.String(),
			"name":       a.name,
			"trainable":  a.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (a *Activation) GetCustomLayerDefinition() string {
	return ``
}

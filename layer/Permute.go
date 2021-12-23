package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Permute struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
	dims      []interface{}
}

func NewPermute(dims []interface{}, options ...PermuteOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		p := &Permute{
			dims:      dims,
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("permute"),
		}
		for _, option := range options {
			option(p)
		}
		return p
	}
}

type PermuteOption func(*Permute)

func PermuteWithName(name string) func(p *Permute) {
	return func(p *Permute) {
		p.name = name
	}
}

func PermuteWithDtype(dtype DataType) func(p *Permute) {
	return func(p *Permute) {
		p.dtype = dtype
	}
}

func PermuteWithTrainable(trainable bool) func(p *Permute) {
	return func(p *Permute) {
		p.trainable = trainable
	}
}

func (p *Permute) GetShape() tf.Shape {
	return p.shape
}

func (p *Permute) GetDtype() DataType {
	return p.dtype
}

func (p *Permute) SetInput(inputs []Layer) {
	p.inputs = inputs
	p.dtype = inputs[0].GetDtype()
}

func (p *Permute) GetInputs() []Layer {
	return p.inputs
}

func (p *Permute) GetName() string {
	return p.name
}

type jsonConfigPermute struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (p *Permute) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range p.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigPermute{
		ClassName: "Permute",
		Name:      p.name,
		Config: map[string]interface{}{
			"dims":      p.dims,
			"dtype":     p.dtype.String(),
			"name":      p.name,
			"trainable": p.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (p *Permute) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Attention struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
	useScale  bool
	causal    bool
	dropout   float64
}

func NewAttention(options ...AttentionOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		a := &Attention{
			useScale:  false,
			causal:    false,
			dropout:   0,
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("attention"),
		}
		for _, option := range options {
			option(a)
		}
		return a
	}
}

type AttentionOption func(*Attention)

func AttentionWithName(name string) func(a *Attention) {
	return func(a *Attention) {
		a.name = name
	}
}

func AttentionWithDtype(dtype DataType) func(a *Attention) {
	return func(a *Attention) {
		a.dtype = dtype
	}
}

func AttentionWithTrainable(trainable bool) func(a *Attention) {
	return func(a *Attention) {
		a.trainable = trainable
	}
}

func AttentionWithUseScale(useScale bool) func(a *Attention) {
	return func(a *Attention) {
		a.useScale = useScale
	}
}

func (a *Attention) GetShape() tf.Shape {
	return a.shape
}

func (a *Attention) GetDtype() DataType {
	return a.dtype
}

func (a *Attention) SetInput(inputs []Layer) {
	a.inputs = inputs
	a.dtype = inputs[0].GetDtype()
}

func (a *Attention) GetInputs() []Layer {
	return a.inputs
}

func (a *Attention) GetName() string {
	return a.name
}

type jsonConfigAttention struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (a *Attention) GetKerasLayerConfig() interface{} {
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
	return jsonConfigAttention{
		ClassName: "Attention",
		Name:      a.name,
		Config: map[string]interface{}{
			"causal":    a.causal,
			"dropout":   a.dropout,
			"dtype":     a.dtype.String(),
			"name":      a.name,
			"trainable": a.trainable,
			"use_scale": a.useScale,
		},
		InboundNodes: inboundNodes,
	}
}

func (a *Attention) GetCustomLayerDefinition() string {
	return ``
}

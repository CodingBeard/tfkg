package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type AdditiveAttention struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
	useScale  bool
	causal    bool
	dropout   float64
}

func NewAdditiveAttention(options ...AdditiveAttentionOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		a := &AdditiveAttention{
			useScale:  true,
			causal:    false,
			dropout:   0,
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("additiveattention"),
		}
		for _, option := range options {
			option(a)
		}
		return a
	}
}

type AdditiveAttentionOption func(*AdditiveAttention)

func AdditiveAttentionWithName(name string) func(a *AdditiveAttention) {
	return func(a *AdditiveAttention) {
		a.name = name
	}
}

func AdditiveAttentionWithDtype(dtype DataType) func(a *AdditiveAttention) {
	return func(a *AdditiveAttention) {
		a.dtype = dtype
	}
}

func AdditiveAttentionWithTrainable(trainable bool) func(a *AdditiveAttention) {
	return func(a *AdditiveAttention) {
		a.trainable = trainable
	}
}

func AdditiveAttentionWithUseScale(useScale bool) func(a *AdditiveAttention) {
	return func(a *AdditiveAttention) {
		a.useScale = useScale
	}
}

func (a *AdditiveAttention) GetShape() tf.Shape {
	return a.shape
}

func (a *AdditiveAttention) GetDtype() DataType {
	return a.dtype
}

func (a *AdditiveAttention) SetInput(inputs []Layer) {
	a.inputs = inputs
	a.dtype = inputs[0].GetDtype()
}

func (a *AdditiveAttention) GetInputs() []Layer {
	return a.inputs
}

func (a *AdditiveAttention) GetName() string {
	return a.name
}

type jsonConfigAdditiveAttention struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (a *AdditiveAttention) GetKerasLayerConfig() interface{} {
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
	return jsonConfigAdditiveAttention{
		ClassName: "AdditiveAttention",
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

func (a *AdditiveAttention) GetCustomLayerDefinition() string {
	return ``
}

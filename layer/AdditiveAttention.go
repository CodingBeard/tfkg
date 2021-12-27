package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LAdditiveAttention struct {
	causal    bool
	dropout   float64
	dtype     DataType
	inputs    []Layer
	name      string
	shape     tf.Shape
	trainable bool
	useScale  bool
}

func AdditiveAttention() *LAdditiveAttention {
	return &LAdditiveAttention{
		causal:    false,
		dropout:   0,
		dtype:     Float32,
		name:      UniqueName("additive_attention"),
		trainable: true,
		useScale:  true,
	}
}

func (l *LAdditiveAttention) SetCausal(causal bool) *LAdditiveAttention {
	l.causal = causal
	return l
}

func (l *LAdditiveAttention) SetDropout(dropout float64) *LAdditiveAttention {
	l.dropout = dropout
	return l
}

func (l *LAdditiveAttention) SetDtype(dtype DataType) *LAdditiveAttention {
	l.dtype = dtype
	return l
}

func (l *LAdditiveAttention) SetName(name string) *LAdditiveAttention {
	l.name = name
	return l
}

func (l *LAdditiveAttention) SetShape(shape tf.Shape) *LAdditiveAttention {
	l.shape = shape
	return l
}

func (l *LAdditiveAttention) SetTrainable(trainable bool) *LAdditiveAttention {
	l.trainable = trainable
	return l
}

func (l *LAdditiveAttention) SetUseScale(useScale bool) *LAdditiveAttention {
	l.useScale = useScale
	return l
}

func (l *LAdditiveAttention) GetShape() tf.Shape {
	return l.shape
}

func (l *LAdditiveAttention) GetDtype() DataType {
	return l.dtype
}

func (l *LAdditiveAttention) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LAdditiveAttention) GetInputs() []Layer {
	return l.inputs
}

func (l *LAdditiveAttention) GetName() string {
	return l.name
}

type jsonConfigLAdditiveAttention struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LAdditiveAttention) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range l.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigLAdditiveAttention{
		ClassName: "AdditiveAttention",
		Name:      l.name,
		Config: map[string]interface{}{
			"causal":    l.causal,
			"dropout":   l.dropout,
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"trainable": l.trainable,
			"use_scale": l.useScale,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LAdditiveAttention) GetCustomLayerDefinition() string {
	return ``
}

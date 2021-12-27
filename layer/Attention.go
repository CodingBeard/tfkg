package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LAttention struct {
	causal    bool
	dropout   float64
	dtype     DataType
	inputs    []Layer
	name      string
	shape     tf.Shape
	trainable bool
	useScale  bool
}

func Attention() *LAttention {
	return &LAttention{
		causal:    false,
		dropout:   0,
		dtype:     Float32,
		name:      UniqueName("attention"),
		trainable: true,
		useScale:  false,
	}
}

func (l *LAttention) SetCausal(causal bool) *LAttention {
	l.causal = causal
	return l
}

func (l *LAttention) SetDropout(dropout float64) *LAttention {
	l.dropout = dropout
	return l
}

func (l *LAttention) SetDtype(dtype DataType) *LAttention {
	l.dtype = dtype
	return l
}

func (l *LAttention) SetName(name string) *LAttention {
	l.name = name
	return l
}

func (l *LAttention) SetShape(shape tf.Shape) *LAttention {
	l.shape = shape
	return l
}

func (l *LAttention) SetTrainable(trainable bool) *LAttention {
	l.trainable = trainable
	return l
}

func (l *LAttention) SetUseScale(useScale bool) *LAttention {
	l.useScale = useScale
	return l
}

func (l *LAttention) GetShape() tf.Shape {
	return l.shape
}

func (l *LAttention) GetDtype() DataType {
	return l.dtype
}

func (l *LAttention) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LAttention) GetInputs() []Layer {
	return l.inputs
}

func (l *LAttention) GetName() string {
	return l.name
}

type jsonConfigLAttention struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LAttention) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLAttention{
		ClassName: "Attention",
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

func (l *LAttention) GetCustomLayerDefinition() string {
	return ``
}

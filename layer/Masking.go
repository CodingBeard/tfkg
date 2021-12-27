package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LMasking struct {
	dtype     DataType
	inputs    []Layer
	maskValue float64
	name      string
	shape     tf.Shape
	trainable bool
}

func Masking() *LMasking {
	return &LMasking{
		dtype:     Float32,
		maskValue: 0,
		name:      UniqueName("masking"),
		trainable: true,
	}
}

func (l *LMasking) SetDtype(dtype DataType) *LMasking {
	l.dtype = dtype
	return l
}

func (l *LMasking) SetMaskValue(maskValue float64) *LMasking {
	l.maskValue = maskValue
	return l
}

func (l *LMasking) SetName(name string) *LMasking {
	l.name = name
	return l
}

func (l *LMasking) SetShape(shape tf.Shape) *LMasking {
	l.shape = shape
	return l
}

func (l *LMasking) SetTrainable(trainable bool) *LMasking {
	l.trainable = trainable
	return l
}

func (l *LMasking) GetShape() tf.Shape {
	return l.shape
}

func (l *LMasking) GetDtype() DataType {
	return l.dtype
}

func (l *LMasking) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LMasking) GetInputs() []Layer {
	return l.inputs
}

func (l *LMasking) GetName() string {
	return l.name
}

type jsonConfigLMasking struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LMasking) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLMasking{
		ClassName: "Masking",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":      l.dtype.String(),
			"mask_value": l.maskValue,
			"name":       l.name,
			"trainable":  l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LMasking) GetCustomLayerDefinition() string {
	return ``
}

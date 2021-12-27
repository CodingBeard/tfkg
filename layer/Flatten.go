package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LFlatten struct {
	dataFormat interface{}
	dtype      DataType
	inputs     []Layer
	name       string
	shape      tf.Shape
	trainable  bool
}

func Flatten() *LFlatten {
	return &LFlatten{
		dataFormat: nil,
		dtype:      Float32,
		name:       UniqueName("flatten"),
		trainable:  true,
	}
}

func (l *LFlatten) SetDataFormat(dataFormat interface{}) *LFlatten {
	l.dataFormat = dataFormat
	return l
}

func (l *LFlatten) SetDtype(dtype DataType) *LFlatten {
	l.dtype = dtype
	return l
}

func (l *LFlatten) SetName(name string) *LFlatten {
	l.name = name
	return l
}

func (l *LFlatten) SetShape(shape tf.Shape) *LFlatten {
	l.shape = shape
	return l
}

func (l *LFlatten) SetTrainable(trainable bool) *LFlatten {
	l.trainable = trainable
	return l
}

func (l *LFlatten) GetShape() tf.Shape {
	return l.shape
}

func (l *LFlatten) GetDtype() DataType {
	return l.dtype
}

func (l *LFlatten) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LFlatten) GetInputs() []Layer {
	return l.inputs
}

func (l *LFlatten) GetName() string {
	return l.name
}

type jsonConfigLFlatten struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LFlatten) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLFlatten{
		ClassName: "Flatten",
		Name:      l.name,
		Config: map[string]interface{}{
			"data_format": l.dataFormat,
			"dtype":       l.dtype.String(),
			"name":        l.name,
			"trainable":   l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LFlatten) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LSubtract struct {
	dtype     DataType
	inputs    []Layer
	name      string
	shape     tf.Shape
	trainable bool
}

func Subtract() *LSubtract {
	return &LSubtract{
		dtype:     Float32,
		name:      UniqueName("subtract"),
		trainable: true,
	}
}

func (l *LSubtract) SetDtype(dtype DataType) *LSubtract {
	l.dtype = dtype
	return l
}

func (l *LSubtract) SetName(name string) *LSubtract {
	l.name = name
	return l
}

func (l *LSubtract) SetShape(shape tf.Shape) *LSubtract {
	l.shape = shape
	return l
}

func (l *LSubtract) SetTrainable(trainable bool) *LSubtract {
	l.trainable = trainable
	return l
}

func (l *LSubtract) GetShape() tf.Shape {
	return l.shape
}

func (l *LSubtract) GetDtype() DataType {
	return l.dtype
}

func (l *LSubtract) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LSubtract) GetInputs() []Layer {
	return l.inputs
}

func (l *LSubtract) GetName() string {
	return l.name
}

type jsonConfigLSubtract struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LSubtract) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLSubtract{
		ClassName: "Subtract",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LSubtract) GetCustomLayerDefinition() string {
	return ``
}

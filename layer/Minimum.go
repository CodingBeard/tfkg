package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LMinimum struct {
	dtype     DataType
	inputs    []Layer
	name      string
	shape     tf.Shape
	trainable bool
}

func Minimum() *LMinimum {
	return &LMinimum{
		dtype:     Float32,
		name:      UniqueName("minimum"),
		trainable: true,
	}
}

func (l *LMinimum) SetDtype(dtype DataType) *LMinimum {
	l.dtype = dtype
	return l
}

func (l *LMinimum) SetName(name string) *LMinimum {
	l.name = name
	return l
}

func (l *LMinimum) SetShape(shape tf.Shape) *LMinimum {
	l.shape = shape
	return l
}

func (l *LMinimum) SetTrainable(trainable bool) *LMinimum {
	l.trainable = trainable
	return l
}

func (l *LMinimum) GetShape() tf.Shape {
	return l.shape
}

func (l *LMinimum) GetDtype() DataType {
	return l.dtype
}

func (l *LMinimum) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LMinimum) GetInputs() []Layer {
	return l.inputs
}

func (l *LMinimum) GetName() string {
	return l.name
}

type jsonConfigLMinimum struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LMinimum) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLMinimum{
		ClassName: "Minimum",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LMinimum) GetCustomLayerDefinition() string {
	return ``
}

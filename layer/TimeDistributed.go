package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LTimeDistributed struct {
	dtype     DataType
	inputs    []Layer
	layer     interface{}
	name      string
	shape     tf.Shape
	trainable bool
}

func TimeDistributed(layer interface{}) *LTimeDistributed {
	return &LTimeDistributed{
		dtype:     Float32,
		layer:     layer,
		name:      UniqueName("time_distributed"),
		trainable: true,
	}
}

func (l *LTimeDistributed) SetDtype(dtype DataType) *LTimeDistributed {
	l.dtype = dtype
	return l
}

func (l *LTimeDistributed) SetName(name string) *LTimeDistributed {
	l.name = name
	return l
}

func (l *LTimeDistributed) SetShape(shape tf.Shape) *LTimeDistributed {
	l.shape = shape
	return l
}

func (l *LTimeDistributed) SetTrainable(trainable bool) *LTimeDistributed {
	l.trainable = trainable
	return l
}

func (l *LTimeDistributed) GetShape() tf.Shape {
	return l.shape
}

func (l *LTimeDistributed) GetDtype() DataType {
	return l.dtype
}

func (l *LTimeDistributed) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LTimeDistributed) GetInputs() []Layer {
	return l.inputs
}

func (l *LTimeDistributed) GetName() string {
	return l.name
}

type jsonConfigLTimeDistributed struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LTimeDistributed) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLTimeDistributed{
		ClassName: "TimeDistributed",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"layer":     l.layer,
			"name":      l.name,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LTimeDistributed) GetCustomLayerDefinition() string {
	return ``
}

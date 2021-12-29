package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LPermute struct {
	dims         []interface{}
	dtype        DataType
	inputs       []Layer
	name         string
	shape        tf.Shape
	trainable    bool
	layerWeights interface{}
}

func Permute(dims []interface{}) *LPermute {
	return &LPermute{
		dims:      dims,
		dtype:     Float32,
		name:      UniqueName("permute"),
		trainable: true,
	}
}

func (l *LPermute) SetDtype(dtype DataType) *LPermute {
	l.dtype = dtype
	return l
}

func (l *LPermute) SetName(name string) *LPermute {
	l.name = name
	return l
}

func (l *LPermute) SetShape(shape tf.Shape) *LPermute {
	l.shape = shape
	return l
}

func (l *LPermute) SetTrainable(trainable bool) *LPermute {
	l.trainable = trainable
	return l
}

func (l *LPermute) SetLayerWeights(layerWeights interface{}) *LPermute {
	l.layerWeights = layerWeights
	return l
}

func (l *LPermute) GetShape() tf.Shape {
	return l.shape
}

func (l *LPermute) GetDtype() DataType {
	return l.dtype
}

func (l *LPermute) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LPermute) GetInputs() []Layer {
	return l.inputs
}

func (l *LPermute) GetName() string {
	return l.name
}

func (l *LPermute) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLPermute struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LPermute) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLPermute{
		ClassName: "Permute",
		Name:      l.name,
		Config: map[string]interface{}{
			"dims":      l.dims,
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LPermute) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LRepeatVector struct {
	dtype        DataType
	inputs       []Layer
	n            float64
	name         string
	shape        tf.Shape
	trainable    bool
	layerWeights interface{}
}

func RepeatVector(n float64) *LRepeatVector {
	return &LRepeatVector{
		dtype:     Float32,
		n:         n,
		name:      UniqueName("repeat_vector"),
		trainable: true,
	}
}

func (l *LRepeatVector) SetDtype(dtype DataType) *LRepeatVector {
	l.dtype = dtype
	return l
}

func (l *LRepeatVector) SetName(name string) *LRepeatVector {
	l.name = name
	return l
}

func (l *LRepeatVector) SetShape(shape tf.Shape) *LRepeatVector {
	l.shape = shape
	return l
}

func (l *LRepeatVector) SetTrainable(trainable bool) *LRepeatVector {
	l.trainable = trainable
	return l
}

func (l *LRepeatVector) SetLayerWeights(layerWeights interface{}) *LRepeatVector {
	l.layerWeights = layerWeights
	return l
}

func (l *LRepeatVector) GetShape() tf.Shape {
	return l.shape
}

func (l *LRepeatVector) GetDtype() DataType {
	return l.dtype
}

func (l *LRepeatVector) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LRepeatVector) GetInputs() []Layer {
	return l.inputs
}

func (l *LRepeatVector) GetName() string {
	return l.name
}

func (l *LRepeatVector) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLRepeatVector struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LRepeatVector) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLRepeatVector{
		ClassName: "RepeatVector",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"n":         l.n,
			"name":      l.name,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LRepeatVector) GetCustomLayerDefinition() string {
	return ``
}

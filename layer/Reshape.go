package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LReshape struct {
	dtype        DataType
	inputs       []Layer
	name         string
	shape        tf.Shape
	targetShape  []interface{}
	trainable    bool
	layerWeights interface{}
}

func Reshape(targetShape []interface{}) *LReshape {
	return &LReshape{
		dtype:       Float32,
		name:        UniqueName("reshape"),
		targetShape: targetShape,
		trainable:   true,
	}
}

func (l *LReshape) SetDtype(dtype DataType) *LReshape {
	l.dtype = dtype
	return l
}

func (l *LReshape) SetName(name string) *LReshape {
	l.name = name
	return l
}

func (l *LReshape) SetShape(shape tf.Shape) *LReshape {
	l.shape = shape
	return l
}

func (l *LReshape) SetTrainable(trainable bool) *LReshape {
	l.trainable = trainable
	return l
}

func (l *LReshape) SetLayerWeights(layerWeights interface{}) *LReshape {
	l.layerWeights = layerWeights
	return l
}

func (l *LReshape) GetShape() tf.Shape {
	return l.shape
}

func (l *LReshape) GetDtype() DataType {
	return l.dtype
}

func (l *LReshape) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LReshape) GetInputs() []Layer {
	return l.inputs
}

func (l *LReshape) GetName() string {
	return l.name
}

func (l *LReshape) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLReshape struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LReshape) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLReshape{
		ClassName: "Reshape",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":        l.dtype.String(),
			"name":         l.name,
			"target_shape": l.targetShape,
			"trainable":    l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LReshape) GetCustomLayerDefinition() string {
	return ``
}

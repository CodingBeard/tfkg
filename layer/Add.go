package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LAdd struct {
	dtype        DataType
	inputs       []Layer
	name         string
	shape        tf.Shape
	trainable    bool
	layerWeights interface{}
}

func Add() *LAdd {
	return &LAdd{
		dtype:     Float32,
		name:      UniqueName("add"),
		trainable: true,
	}
}

func (l *LAdd) SetDtype(dtype DataType) *LAdd {
	l.dtype = dtype
	return l
}

func (l *LAdd) SetName(name string) *LAdd {
	l.name = name
	return l
}

func (l *LAdd) SetShape(shape tf.Shape) *LAdd {
	l.shape = shape
	return l
}

func (l *LAdd) SetTrainable(trainable bool) *LAdd {
	l.trainable = trainable
	return l
}

func (l *LAdd) SetLayerWeights(layerWeights interface{}) *LAdd {
	l.layerWeights = layerWeights
	return l
}

func (l *LAdd) GetShape() tf.Shape {
	return l.shape
}

func (l *LAdd) GetDtype() DataType {
	return l.dtype
}

func (l *LAdd) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LAdd) GetInputs() []Layer {
	return l.inputs
}

func (l *LAdd) GetName() string {
	return l.name
}

func (l *LAdd) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLAdd struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LAdd) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLAdd{
		ClassName: "Add",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LAdd) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LMultiply struct {
	dtype        DataType
	inputs       []Layer
	name         string
	shape        tf.Shape
	trainable    bool
	layerWeights []*tf.Tensor
}

func Multiply() *LMultiply {
	return &LMultiply{
		dtype:     Float32,
		name:      UniqueName("multiply"),
		trainable: true,
	}
}

func (l *LMultiply) SetDtype(dtype DataType) *LMultiply {
	l.dtype = dtype
	return l
}

func (l *LMultiply) SetName(name string) *LMultiply {
	l.name = name
	return l
}

func (l *LMultiply) SetShape(shape tf.Shape) *LMultiply {
	l.shape = shape
	return l
}

func (l *LMultiply) SetTrainable(trainable bool) *LMultiply {
	l.trainable = trainable
	return l
}

func (l *LMultiply) SetLayerWeights(layerWeights []*tf.Tensor) *LMultiply {
	l.layerWeights = layerWeights
	return l
}

func (l *LMultiply) GetShape() tf.Shape {
	return l.shape
}

func (l *LMultiply) GetDtype() DataType {
	return l.dtype
}

func (l *LMultiply) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LMultiply) GetInputs() []Layer {
	return l.inputs
}

func (l *LMultiply) GetName() string {
	return l.name
}

func (l *LMultiply) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLMultiply struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LMultiply) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLMultiply{
		ClassName: "Multiply",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LMultiply) GetCustomLayerDefinition() string {
	return ``
}

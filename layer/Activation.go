package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LActivation struct {
	activation   string
	dtype        DataType
	inputs       []Layer
	name         string
	shape        tf.Shape
	trainable    bool
	layerWeights []*tf.Tensor
}

func Activation(activation string) *LActivation {
	return &LActivation{
		activation: activation,
		dtype:      Float32,
		name:       UniqueName("activation"),
		trainable:  true,
	}
}

func (l *LActivation) SetDtype(dtype DataType) *LActivation {
	l.dtype = dtype
	return l
}

func (l *LActivation) SetName(name string) *LActivation {
	l.name = name
	return l
}

func (l *LActivation) SetShape(shape tf.Shape) *LActivation {
	l.shape = shape
	return l
}

func (l *LActivation) SetTrainable(trainable bool) *LActivation {
	l.trainable = trainable
	return l
}

func (l *LActivation) SetLayerWeights(layerWeights []*tf.Tensor) *LActivation {
	l.layerWeights = layerWeights
	return l
}

func (l *LActivation) GetShape() tf.Shape {
	return l.shape
}

func (l *LActivation) GetDtype() DataType {
	return l.dtype
}

func (l *LActivation) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LActivation) GetInputs() []Layer {
	return l.inputs
}

func (l *LActivation) GetName() string {
	return l.name
}

func (l *LActivation) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLActivation struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LActivation) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLActivation{
		ClassName: "Activation",
		Name:      l.name,
		Config: map[string]interface{}{
			"activation": l.activation,
			"dtype":      l.dtype.String(),
			"name":       l.name,
			"trainable":  l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LActivation) GetCustomLayerDefinition() string {
	return ``
}

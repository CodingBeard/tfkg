package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LLeakyReLU struct {
	alpha        float64
	dtype        DataType
	inputs       []Layer
	name         string
	shape        tf.Shape
	trainable    bool
	layerWeights []*tf.Tensor
}

func LeakyReLU() *LLeakyReLU {
	return &LLeakyReLU{
		alpha:     0.3,
		dtype:     Float32,
		name:      UniqueName("leaky_re_lu"),
		trainable: true,
	}
}

func (l *LLeakyReLU) SetAlpha(alpha float64) *LLeakyReLU {
	l.alpha = alpha
	return l
}

func (l *LLeakyReLU) SetDtype(dtype DataType) *LLeakyReLU {
	l.dtype = dtype
	return l
}

func (l *LLeakyReLU) SetName(name string) *LLeakyReLU {
	l.name = name
	return l
}

func (l *LLeakyReLU) SetShape(shape tf.Shape) *LLeakyReLU {
	l.shape = shape
	return l
}

func (l *LLeakyReLU) SetTrainable(trainable bool) *LLeakyReLU {
	l.trainable = trainable
	return l
}

func (l *LLeakyReLU) SetLayerWeights(layerWeights []*tf.Tensor) *LLeakyReLU {
	l.layerWeights = layerWeights
	return l
}

func (l *LLeakyReLU) GetShape() tf.Shape {
	return l.shape
}

func (l *LLeakyReLU) GetDtype() DataType {
	return l.dtype
}

func (l *LLeakyReLU) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LLeakyReLU) GetInputs() []Layer {
	return l.inputs
}

func (l *LLeakyReLU) GetName() string {
	return l.name
}

func (l *LLeakyReLU) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLLeakyReLU struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LLeakyReLU) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLLeakyReLU{
		ClassName: "LeakyReLU",
		Name:      l.name,
		Config: map[string]interface{}{
			"alpha":     l.alpha,
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LLeakyReLU) GetCustomLayerDefinition() string {
	return ``
}

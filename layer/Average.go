package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LAverage struct {
	dtype        DataType
	inputs       []Layer
	name         string
	shape        tf.Shape
	trainable    bool
	layerWeights []*tf.Tensor
}

func Average() *LAverage {
	return &LAverage{
		dtype:     Float32,
		name:      UniqueName("average"),
		trainable: true,
	}
}

func (l *LAverage) SetDtype(dtype DataType) *LAverage {
	l.dtype = dtype
	return l
}

func (l *LAverage) SetName(name string) *LAverage {
	l.name = name
	return l
}

func (l *LAverage) SetShape(shape tf.Shape) *LAverage {
	l.shape = shape
	return l
}

func (l *LAverage) SetTrainable(trainable bool) *LAverage {
	l.trainable = trainable
	return l
}

func (l *LAverage) SetLayerWeights(layerWeights []*tf.Tensor) *LAverage {
	l.layerWeights = layerWeights
	return l
}

func (l *LAverage) GetShape() tf.Shape {
	return l.shape
}

func (l *LAverage) GetDtype() DataType {
	return l.dtype
}

func (l *LAverage) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LAverage) GetInputs() []Layer {
	return l.inputs
}

func (l *LAverage) GetName() string {
	return l.name
}

func (l *LAverage) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLAverage struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LAverage) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLAverage{
		ClassName: "Average",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LAverage) GetCustomLayerDefinition() string {
	return ``
}

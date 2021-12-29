package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LPreprocessingLayer struct {
	dtype        DataType
	inputs       []Layer
	name         string
	shape        tf.Shape
	trainable    bool
	layerWeights interface{}
}

func PreprocessingLayer() *LPreprocessingLayer {
	return &LPreprocessingLayer{
		dtype:     Float32,
		name:      UniqueName("preprocessing_layer"),
		trainable: true,
	}
}

func (l *LPreprocessingLayer) SetDtype(dtype DataType) *LPreprocessingLayer {
	l.dtype = dtype
	return l
}

func (l *LPreprocessingLayer) SetName(name string) *LPreprocessingLayer {
	l.name = name
	return l
}

func (l *LPreprocessingLayer) SetShape(shape tf.Shape) *LPreprocessingLayer {
	l.shape = shape
	return l
}

func (l *LPreprocessingLayer) SetTrainable(trainable bool) *LPreprocessingLayer {
	l.trainable = trainable
	return l
}

func (l *LPreprocessingLayer) SetLayerWeights(layerWeights interface{}) *LPreprocessingLayer {
	l.layerWeights = layerWeights
	return l
}

func (l *LPreprocessingLayer) GetShape() tf.Shape {
	return l.shape
}

func (l *LPreprocessingLayer) GetDtype() DataType {
	return l.dtype
}

func (l *LPreprocessingLayer) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LPreprocessingLayer) GetInputs() []Layer {
	return l.inputs
}

func (l *LPreprocessingLayer) GetName() string {
	return l.name
}

func (l *LPreprocessingLayer) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLPreprocessingLayer struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LPreprocessingLayer) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLPreprocessingLayer{
		ClassName: "PreprocessingLayer",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LPreprocessingLayer) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LGlobalAveragePooling2D struct {
	dataFormat   interface{}
	dtype        DataType
	inputs       []Layer
	keepdims     bool
	name         string
	shape        tf.Shape
	trainable    bool
	layerWeights interface{}
}

func GlobalAveragePooling2D() *LGlobalAveragePooling2D {
	return &LGlobalAveragePooling2D{
		dataFormat: nil,
		dtype:      Float32,
		keepdims:   false,
		name:       UniqueName("global_average_pooling2d"),
		trainable:  true,
	}
}

func (l *LGlobalAveragePooling2D) SetDataFormat(dataFormat interface{}) *LGlobalAveragePooling2D {
	l.dataFormat = dataFormat
	return l
}

func (l *LGlobalAveragePooling2D) SetDtype(dtype DataType) *LGlobalAveragePooling2D {
	l.dtype = dtype
	return l
}

func (l *LGlobalAveragePooling2D) SetKeepdims(keepdims bool) *LGlobalAveragePooling2D {
	l.keepdims = keepdims
	return l
}

func (l *LGlobalAveragePooling2D) SetName(name string) *LGlobalAveragePooling2D {
	l.name = name
	return l
}

func (l *LGlobalAveragePooling2D) SetShape(shape tf.Shape) *LGlobalAveragePooling2D {
	l.shape = shape
	return l
}

func (l *LGlobalAveragePooling2D) SetTrainable(trainable bool) *LGlobalAveragePooling2D {
	l.trainable = trainable
	return l
}

func (l *LGlobalAveragePooling2D) SetLayerWeights(layerWeights interface{}) *LGlobalAveragePooling2D {
	l.layerWeights = layerWeights
	return l
}

func (l *LGlobalAveragePooling2D) GetShape() tf.Shape {
	return l.shape
}

func (l *LGlobalAveragePooling2D) GetDtype() DataType {
	return l.dtype
}

func (l *LGlobalAveragePooling2D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LGlobalAveragePooling2D) GetInputs() []Layer {
	return l.inputs
}

func (l *LGlobalAveragePooling2D) GetName() string {
	return l.name
}

func (l *LGlobalAveragePooling2D) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLGlobalAveragePooling2D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LGlobalAveragePooling2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLGlobalAveragePooling2D{
		ClassName: "GlobalAveragePooling2D",
		Name:      l.name,
		Config: map[string]interface{}{
			"data_format": l.dataFormat,
			"dtype":       l.dtype.String(),
			"keepdims":    l.keepdims,
			"name":        l.name,
			"trainable":   l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LGlobalAveragePooling2D) GetCustomLayerDefinition() string {
	return ``
}

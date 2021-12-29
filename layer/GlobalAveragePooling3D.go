package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LGlobalAveragePooling3D struct {
	dataFormat   interface{}
	dtype        DataType
	inputs       []Layer
	keepdims     bool
	name         string
	shape        tf.Shape
	trainable    bool
	layerWeights interface{}
}

func GlobalAveragePooling3D() *LGlobalAveragePooling3D {
	return &LGlobalAveragePooling3D{
		dataFormat: nil,
		dtype:      Float32,
		keepdims:   false,
		name:       UniqueName("global_average_pooling3d"),
		trainable:  true,
	}
}

func (l *LGlobalAveragePooling3D) SetDataFormat(dataFormat interface{}) *LGlobalAveragePooling3D {
	l.dataFormat = dataFormat
	return l
}

func (l *LGlobalAveragePooling3D) SetDtype(dtype DataType) *LGlobalAveragePooling3D {
	l.dtype = dtype
	return l
}

func (l *LGlobalAveragePooling3D) SetKeepdims(keepdims bool) *LGlobalAveragePooling3D {
	l.keepdims = keepdims
	return l
}

func (l *LGlobalAveragePooling3D) SetName(name string) *LGlobalAveragePooling3D {
	l.name = name
	return l
}

func (l *LGlobalAveragePooling3D) SetShape(shape tf.Shape) *LGlobalAveragePooling3D {
	l.shape = shape
	return l
}

func (l *LGlobalAveragePooling3D) SetTrainable(trainable bool) *LGlobalAveragePooling3D {
	l.trainable = trainable
	return l
}

func (l *LGlobalAveragePooling3D) SetLayerWeights(layerWeights interface{}) *LGlobalAveragePooling3D {
	l.layerWeights = layerWeights
	return l
}

func (l *LGlobalAveragePooling3D) GetShape() tf.Shape {
	return l.shape
}

func (l *LGlobalAveragePooling3D) GetDtype() DataType {
	return l.dtype
}

func (l *LGlobalAveragePooling3D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LGlobalAveragePooling3D) GetInputs() []Layer {
	return l.inputs
}

func (l *LGlobalAveragePooling3D) GetName() string {
	return l.name
}

func (l *LGlobalAveragePooling3D) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLGlobalAveragePooling3D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LGlobalAveragePooling3D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLGlobalAveragePooling3D{
		ClassName: "GlobalAveragePooling3D",
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

func (l *LGlobalAveragePooling3D) GetCustomLayerDefinition() string {
	return ``
}

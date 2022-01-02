package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LGlobalMaxPooling3D struct {
	dataFormat   interface{}
	dtype        DataType
	inputs       []Layer
	keepdims     bool
	name         string
	shape        tf.Shape
	trainable    bool
	layerWeights []*tf.Tensor
}

func GlobalMaxPooling3D() *LGlobalMaxPooling3D {
	return &LGlobalMaxPooling3D{
		dataFormat: nil,
		dtype:      Float32,
		keepdims:   false,
		name:       UniqueName("global_max_pooling3d"),
		trainable:  true,
	}
}

func (l *LGlobalMaxPooling3D) SetDataFormat(dataFormat interface{}) *LGlobalMaxPooling3D {
	l.dataFormat = dataFormat
	return l
}

func (l *LGlobalMaxPooling3D) SetDtype(dtype DataType) *LGlobalMaxPooling3D {
	l.dtype = dtype
	return l
}

func (l *LGlobalMaxPooling3D) SetKeepdims(keepdims bool) *LGlobalMaxPooling3D {
	l.keepdims = keepdims
	return l
}

func (l *LGlobalMaxPooling3D) SetName(name string) *LGlobalMaxPooling3D {
	l.name = name
	return l
}

func (l *LGlobalMaxPooling3D) SetShape(shape tf.Shape) *LGlobalMaxPooling3D {
	l.shape = shape
	return l
}

func (l *LGlobalMaxPooling3D) SetTrainable(trainable bool) *LGlobalMaxPooling3D {
	l.trainable = trainable
	return l
}

func (l *LGlobalMaxPooling3D) SetLayerWeights(layerWeights []*tf.Tensor) *LGlobalMaxPooling3D {
	l.layerWeights = layerWeights
	return l
}

func (l *LGlobalMaxPooling3D) GetShape() tf.Shape {
	return l.shape
}

func (l *LGlobalMaxPooling3D) GetDtype() DataType {
	return l.dtype
}

func (l *LGlobalMaxPooling3D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LGlobalMaxPooling3D) GetInputs() []Layer {
	return l.inputs
}

func (l *LGlobalMaxPooling3D) GetName() string {
	return l.name
}

func (l *LGlobalMaxPooling3D) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLGlobalMaxPooling3D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LGlobalMaxPooling3D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLGlobalMaxPooling3D{
		ClassName: "GlobalMaxPooling3D",
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

func (l *LGlobalMaxPooling3D) GetCustomLayerDefinition() string {
	return ``
}

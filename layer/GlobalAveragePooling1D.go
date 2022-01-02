package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LGlobalAveragePooling1D struct {
	dataFormat   string
	dtype        DataType
	inputs       []Layer
	keepdims     bool
	name         string
	shape        tf.Shape
	trainable    bool
	layerWeights []*tf.Tensor
}

func GlobalAveragePooling1D() *LGlobalAveragePooling1D {
	return &LGlobalAveragePooling1D{
		dataFormat: "channels_last",
		dtype:      Float32,
		keepdims:   false,
		name:       UniqueName("global_average_pooling1d"),
		trainable:  true,
	}
}

func (l *LGlobalAveragePooling1D) SetDataFormat(dataFormat string) *LGlobalAveragePooling1D {
	l.dataFormat = dataFormat
	return l
}

func (l *LGlobalAveragePooling1D) SetDtype(dtype DataType) *LGlobalAveragePooling1D {
	l.dtype = dtype
	return l
}

func (l *LGlobalAveragePooling1D) SetKeepdims(keepdims bool) *LGlobalAveragePooling1D {
	l.keepdims = keepdims
	return l
}

func (l *LGlobalAveragePooling1D) SetName(name string) *LGlobalAveragePooling1D {
	l.name = name
	return l
}

func (l *LGlobalAveragePooling1D) SetShape(shape tf.Shape) *LGlobalAveragePooling1D {
	l.shape = shape
	return l
}

func (l *LGlobalAveragePooling1D) SetTrainable(trainable bool) *LGlobalAveragePooling1D {
	l.trainable = trainable
	return l
}

func (l *LGlobalAveragePooling1D) SetLayerWeights(layerWeights []*tf.Tensor) *LGlobalAveragePooling1D {
	l.layerWeights = layerWeights
	return l
}

func (l *LGlobalAveragePooling1D) GetShape() tf.Shape {
	return l.shape
}

func (l *LGlobalAveragePooling1D) GetDtype() DataType {
	return l.dtype
}

func (l *LGlobalAveragePooling1D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LGlobalAveragePooling1D) GetInputs() []Layer {
	return l.inputs
}

func (l *LGlobalAveragePooling1D) GetName() string {
	return l.name
}

func (l *LGlobalAveragePooling1D) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLGlobalAveragePooling1D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LGlobalAveragePooling1D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLGlobalAveragePooling1D{
		ClassName: "GlobalAveragePooling1D",
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

func (l *LGlobalAveragePooling1D) GetCustomLayerDefinition() string {
	return ``
}

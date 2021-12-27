package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LGlobalMaxPooling1D struct {
	dataFormat string
	dtype      DataType
	inputs     []Layer
	keepdims   bool
	name       string
	shape      tf.Shape
	trainable  bool
}

func GlobalMaxPooling1D() *LGlobalMaxPooling1D {
	return &LGlobalMaxPooling1D{
		dataFormat: "channels_last",
		dtype:      Float32,
		keepdims:   false,
		name:       UniqueName("global_max_pooling1d"),
		trainable:  true,
	}
}

func (l *LGlobalMaxPooling1D) SetDataFormat(dataFormat string) *LGlobalMaxPooling1D {
	l.dataFormat = dataFormat
	return l
}

func (l *LGlobalMaxPooling1D) SetDtype(dtype DataType) *LGlobalMaxPooling1D {
	l.dtype = dtype
	return l
}

func (l *LGlobalMaxPooling1D) SetKeepdims(keepdims bool) *LGlobalMaxPooling1D {
	l.keepdims = keepdims
	return l
}

func (l *LGlobalMaxPooling1D) SetName(name string) *LGlobalMaxPooling1D {
	l.name = name
	return l
}

func (l *LGlobalMaxPooling1D) SetShape(shape tf.Shape) *LGlobalMaxPooling1D {
	l.shape = shape
	return l
}

func (l *LGlobalMaxPooling1D) SetTrainable(trainable bool) *LGlobalMaxPooling1D {
	l.trainable = trainable
	return l
}

func (l *LGlobalMaxPooling1D) GetShape() tf.Shape {
	return l.shape
}

func (l *LGlobalMaxPooling1D) GetDtype() DataType {
	return l.dtype
}

func (l *LGlobalMaxPooling1D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LGlobalMaxPooling1D) GetInputs() []Layer {
	return l.inputs
}

func (l *LGlobalMaxPooling1D) GetName() string {
	return l.name
}

type jsonConfigLGlobalMaxPooling1D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LGlobalMaxPooling1D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLGlobalMaxPooling1D{
		ClassName: "GlobalMaxPooling1D",
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

func (l *LGlobalMaxPooling1D) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LGlobalMaxPooling2D struct {
	dataFormat interface{}
	dtype      DataType
	inputs     []Layer
	keepdims   bool
	name       string
	shape      tf.Shape
	trainable  bool
}

func GlobalMaxPooling2D() *LGlobalMaxPooling2D {
	return &LGlobalMaxPooling2D{
		dataFormat: nil,
		dtype:      Float32,
		keepdims:   false,
		name:       UniqueName("global_max_pooling2d"),
		trainable:  true,
	}
}

func (l *LGlobalMaxPooling2D) SetDataFormat(dataFormat interface{}) *LGlobalMaxPooling2D {
	l.dataFormat = dataFormat
	return l
}

func (l *LGlobalMaxPooling2D) SetDtype(dtype DataType) *LGlobalMaxPooling2D {
	l.dtype = dtype
	return l
}

func (l *LGlobalMaxPooling2D) SetKeepdims(keepdims bool) *LGlobalMaxPooling2D {
	l.keepdims = keepdims
	return l
}

func (l *LGlobalMaxPooling2D) SetName(name string) *LGlobalMaxPooling2D {
	l.name = name
	return l
}

func (l *LGlobalMaxPooling2D) SetShape(shape tf.Shape) *LGlobalMaxPooling2D {
	l.shape = shape
	return l
}

func (l *LGlobalMaxPooling2D) SetTrainable(trainable bool) *LGlobalMaxPooling2D {
	l.trainable = trainable
	return l
}

func (l *LGlobalMaxPooling2D) GetShape() tf.Shape {
	return l.shape
}

func (l *LGlobalMaxPooling2D) GetDtype() DataType {
	return l.dtype
}

func (l *LGlobalMaxPooling2D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LGlobalMaxPooling2D) GetInputs() []Layer {
	return l.inputs
}

func (l *LGlobalMaxPooling2D) GetName() string {
	return l.name
}

type jsonConfigLGlobalMaxPooling2D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LGlobalMaxPooling2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLGlobalMaxPooling2D{
		ClassName: "GlobalMaxPooling2D",
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

func (l *LGlobalMaxPooling2D) GetCustomLayerDefinition() string {
	return ``
}

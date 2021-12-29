package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LMaxPooling1D struct {
	dataFormat   string
	dtype        DataType
	inputs       []Layer
	name         string
	padding      string
	poolSize     float64
	shape        tf.Shape
	strides      interface{}
	trainable    bool
	layerWeights interface{}
}

func MaxPooling1D() *LMaxPooling1D {
	return &LMaxPooling1D{
		dataFormat: "channels_last",
		dtype:      Float32,
		name:       UniqueName("max_pooling1d"),
		padding:    "valid",
		poolSize:   2,
		strides:    nil,
		trainable:  true,
	}
}

func (l *LMaxPooling1D) SetDataFormat(dataFormat string) *LMaxPooling1D {
	l.dataFormat = dataFormat
	return l
}

func (l *LMaxPooling1D) SetDtype(dtype DataType) *LMaxPooling1D {
	l.dtype = dtype
	return l
}

func (l *LMaxPooling1D) SetName(name string) *LMaxPooling1D {
	l.name = name
	return l
}

func (l *LMaxPooling1D) SetPadding(padding string) *LMaxPooling1D {
	l.padding = padding
	return l
}

func (l *LMaxPooling1D) SetPoolSize(poolSize float64) *LMaxPooling1D {
	l.poolSize = poolSize
	return l
}

func (l *LMaxPooling1D) SetShape(shape tf.Shape) *LMaxPooling1D {
	l.shape = shape
	return l
}

func (l *LMaxPooling1D) SetStrides(strides interface{}) *LMaxPooling1D {
	l.strides = strides
	return l
}

func (l *LMaxPooling1D) SetTrainable(trainable bool) *LMaxPooling1D {
	l.trainable = trainable
	return l
}

func (l *LMaxPooling1D) SetLayerWeights(layerWeights interface{}) *LMaxPooling1D {
	l.layerWeights = layerWeights
	return l
}

func (l *LMaxPooling1D) GetShape() tf.Shape {
	return l.shape
}

func (l *LMaxPooling1D) GetDtype() DataType {
	return l.dtype
}

func (l *LMaxPooling1D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LMaxPooling1D) GetInputs() []Layer {
	return l.inputs
}

func (l *LMaxPooling1D) GetName() string {
	return l.name
}

func (l *LMaxPooling1D) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLMaxPooling1D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LMaxPooling1D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLMaxPooling1D{
		ClassName: "MaxPooling1D",
		Name:      l.name,
		Config: map[string]interface{}{
			"data_format": l.dataFormat,
			"dtype":       l.dtype.String(),
			"name":        l.name,
			"padding":     l.padding,
			"pool_size":   l.poolSize,
			"strides":     l.strides,
			"trainable":   l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LMaxPooling1D) GetCustomLayerDefinition() string {
	return ``
}

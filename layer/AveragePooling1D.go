package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LAveragePooling1D struct {
	dataFormat   string
	dtype        DataType
	inputs       []Layer
	name         string
	padding      string
	poolSize     float64
	shape        tf.Shape
	strides      interface{}
	trainable    bool
	layerWeights []*tf.Tensor
}

func AveragePooling1D() *LAveragePooling1D {
	return &LAveragePooling1D{
		dataFormat: "channels_last",
		dtype:      Float32,
		name:       UniqueName("average_pooling1d"),
		padding:    "valid",
		poolSize:   2,
		strides:    nil,
		trainable:  true,
	}
}

func (l *LAveragePooling1D) SetDataFormat(dataFormat string) *LAveragePooling1D {
	l.dataFormat = dataFormat
	return l
}

func (l *LAveragePooling1D) SetDtype(dtype DataType) *LAveragePooling1D {
	l.dtype = dtype
	return l
}

func (l *LAveragePooling1D) SetName(name string) *LAveragePooling1D {
	l.name = name
	return l
}

func (l *LAveragePooling1D) SetPadding(padding string) *LAveragePooling1D {
	l.padding = padding
	return l
}

func (l *LAveragePooling1D) SetPoolSize(poolSize float64) *LAveragePooling1D {
	l.poolSize = poolSize
	return l
}

func (l *LAveragePooling1D) SetShape(shape tf.Shape) *LAveragePooling1D {
	l.shape = shape
	return l
}

func (l *LAveragePooling1D) SetStrides(strides interface{}) *LAveragePooling1D {
	l.strides = strides
	return l
}

func (l *LAveragePooling1D) SetTrainable(trainable bool) *LAveragePooling1D {
	l.trainable = trainable
	return l
}

func (l *LAveragePooling1D) SetLayerWeights(layerWeights []*tf.Tensor) *LAveragePooling1D {
	l.layerWeights = layerWeights
	return l
}

func (l *LAveragePooling1D) GetShape() tf.Shape {
	return l.shape
}

func (l *LAveragePooling1D) GetDtype() DataType {
	return l.dtype
}

func (l *LAveragePooling1D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LAveragePooling1D) GetInputs() []Layer {
	return l.inputs
}

func (l *LAveragePooling1D) GetName() string {
	return l.name
}

func (l *LAveragePooling1D) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLAveragePooling1D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LAveragePooling1D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLAveragePooling1D{
		ClassName: "AveragePooling1D",
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

func (l *LAveragePooling1D) GetCustomLayerDefinition() string {
	return ``
}

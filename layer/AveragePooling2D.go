package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LAveragePooling2D struct {
	dataFormat   interface{}
	dtype        DataType
	inputs       []Layer
	name         string
	padding      string
	poolSize     []interface{}
	shape        tf.Shape
	strides      interface{}
	trainable    bool
	layerWeights interface{}
}

func AveragePooling2D() *LAveragePooling2D {
	return &LAveragePooling2D{
		dataFormat: nil,
		dtype:      Float32,
		name:       UniqueName("average_pooling2d"),
		padding:    "valid",
		poolSize:   []interface{}{2, 2},
		strides:    nil,
		trainable:  true,
	}
}

func (l *LAveragePooling2D) SetDataFormat(dataFormat interface{}) *LAveragePooling2D {
	l.dataFormat = dataFormat
	return l
}

func (l *LAveragePooling2D) SetDtype(dtype DataType) *LAveragePooling2D {
	l.dtype = dtype
	return l
}

func (l *LAveragePooling2D) SetName(name string) *LAveragePooling2D {
	l.name = name
	return l
}

func (l *LAveragePooling2D) SetPadding(padding string) *LAveragePooling2D {
	l.padding = padding
	return l
}

func (l *LAveragePooling2D) SetPoolSize(poolSize []interface{}) *LAveragePooling2D {
	l.poolSize = poolSize
	return l
}

func (l *LAveragePooling2D) SetShape(shape tf.Shape) *LAveragePooling2D {
	l.shape = shape
	return l
}

func (l *LAveragePooling2D) SetStrides(strides interface{}) *LAveragePooling2D {
	l.strides = strides
	return l
}

func (l *LAveragePooling2D) SetTrainable(trainable bool) *LAveragePooling2D {
	l.trainable = trainable
	return l
}

func (l *LAveragePooling2D) SetLayerWeights(layerWeights interface{}) *LAveragePooling2D {
	l.layerWeights = layerWeights
	return l
}

func (l *LAveragePooling2D) GetShape() tf.Shape {
	return l.shape
}

func (l *LAveragePooling2D) GetDtype() DataType {
	return l.dtype
}

func (l *LAveragePooling2D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LAveragePooling2D) GetInputs() []Layer {
	return l.inputs
}

func (l *LAveragePooling2D) GetName() string {
	return l.name
}

func (l *LAveragePooling2D) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLAveragePooling2D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LAveragePooling2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLAveragePooling2D{
		ClassName: "AveragePooling2D",
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

func (l *LAveragePooling2D) GetCustomLayerDefinition() string {
	return ``
}

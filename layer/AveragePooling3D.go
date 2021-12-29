package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LAveragePooling3D struct {
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

func AveragePooling3D() *LAveragePooling3D {
	return &LAveragePooling3D{
		dataFormat: nil,
		dtype:      Float32,
		name:       UniqueName("average_pooling3d"),
		padding:    "valid",
		poolSize:   []interface{}{2, 2, 2},
		strides:    nil,
		trainable:  true,
	}
}

func (l *LAveragePooling3D) SetDataFormat(dataFormat interface{}) *LAveragePooling3D {
	l.dataFormat = dataFormat
	return l
}

func (l *LAveragePooling3D) SetDtype(dtype DataType) *LAveragePooling3D {
	l.dtype = dtype
	return l
}

func (l *LAveragePooling3D) SetName(name string) *LAveragePooling3D {
	l.name = name
	return l
}

func (l *LAveragePooling3D) SetPadding(padding string) *LAveragePooling3D {
	l.padding = padding
	return l
}

func (l *LAveragePooling3D) SetPoolSize(poolSize []interface{}) *LAveragePooling3D {
	l.poolSize = poolSize
	return l
}

func (l *LAveragePooling3D) SetShape(shape tf.Shape) *LAveragePooling3D {
	l.shape = shape
	return l
}

func (l *LAveragePooling3D) SetStrides(strides interface{}) *LAveragePooling3D {
	l.strides = strides
	return l
}

func (l *LAveragePooling3D) SetTrainable(trainable bool) *LAveragePooling3D {
	l.trainable = trainable
	return l
}

func (l *LAveragePooling3D) SetLayerWeights(layerWeights interface{}) *LAveragePooling3D {
	l.layerWeights = layerWeights
	return l
}

func (l *LAveragePooling3D) GetShape() tf.Shape {
	return l.shape
}

func (l *LAveragePooling3D) GetDtype() DataType {
	return l.dtype
}

func (l *LAveragePooling3D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LAveragePooling3D) GetInputs() []Layer {
	return l.inputs
}

func (l *LAveragePooling3D) GetName() string {
	return l.name
}

func (l *LAveragePooling3D) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLAveragePooling3D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LAveragePooling3D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLAveragePooling3D{
		ClassName: "AveragePooling3D",
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

func (l *LAveragePooling3D) GetCustomLayerDefinition() string {
	return ``
}

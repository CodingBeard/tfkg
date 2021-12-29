package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LMaxPooling2D struct {
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

func MaxPooling2D() *LMaxPooling2D {
	return &LMaxPooling2D{
		dataFormat: nil,
		dtype:      Float32,
		name:       UniqueName("max_pooling2d"),
		padding:    "valid",
		poolSize:   []interface{}{2, 2},
		strides:    nil,
		trainable:  true,
	}
}

func (l *LMaxPooling2D) SetDataFormat(dataFormat interface{}) *LMaxPooling2D {
	l.dataFormat = dataFormat
	return l
}

func (l *LMaxPooling2D) SetDtype(dtype DataType) *LMaxPooling2D {
	l.dtype = dtype
	return l
}

func (l *LMaxPooling2D) SetName(name string) *LMaxPooling2D {
	l.name = name
	return l
}

func (l *LMaxPooling2D) SetPadding(padding string) *LMaxPooling2D {
	l.padding = padding
	return l
}

func (l *LMaxPooling2D) SetPoolSize(poolSize []interface{}) *LMaxPooling2D {
	l.poolSize = poolSize
	return l
}

func (l *LMaxPooling2D) SetShape(shape tf.Shape) *LMaxPooling2D {
	l.shape = shape
	return l
}

func (l *LMaxPooling2D) SetStrides(strides interface{}) *LMaxPooling2D {
	l.strides = strides
	return l
}

func (l *LMaxPooling2D) SetTrainable(trainable bool) *LMaxPooling2D {
	l.trainable = trainable
	return l
}

func (l *LMaxPooling2D) SetLayerWeights(layerWeights interface{}) *LMaxPooling2D {
	l.layerWeights = layerWeights
	return l
}

func (l *LMaxPooling2D) GetShape() tf.Shape {
	return l.shape
}

func (l *LMaxPooling2D) GetDtype() DataType {
	return l.dtype
}

func (l *LMaxPooling2D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LMaxPooling2D) GetInputs() []Layer {
	return l.inputs
}

func (l *LMaxPooling2D) GetName() string {
	return l.name
}

func (l *LMaxPooling2D) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLMaxPooling2D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LMaxPooling2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLMaxPooling2D{
		ClassName: "MaxPooling2D",
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

func (l *LMaxPooling2D) GetCustomLayerDefinition() string {
	return ``
}

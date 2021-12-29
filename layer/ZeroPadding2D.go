package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LZeroPadding2D struct {
	dataFormat   interface{}
	dtype        DataType
	inputs       []Layer
	name         string
	padding      []interface{}
	shape        tf.Shape
	trainable    bool
	layerWeights interface{}
}

func ZeroPadding2D() *LZeroPadding2D {
	return &LZeroPadding2D{
		dataFormat: nil,
		dtype:      Float32,
		name:       UniqueName("zero_padding2d"),
		padding:    []interface{}{1, 1},
		trainable:  true,
	}
}

func (l *LZeroPadding2D) SetDataFormat(dataFormat interface{}) *LZeroPadding2D {
	l.dataFormat = dataFormat
	return l
}

func (l *LZeroPadding2D) SetDtype(dtype DataType) *LZeroPadding2D {
	l.dtype = dtype
	return l
}

func (l *LZeroPadding2D) SetName(name string) *LZeroPadding2D {
	l.name = name
	return l
}

func (l *LZeroPadding2D) SetPadding(padding []interface{}) *LZeroPadding2D {
	l.padding = padding
	return l
}

func (l *LZeroPadding2D) SetShape(shape tf.Shape) *LZeroPadding2D {
	l.shape = shape
	return l
}

func (l *LZeroPadding2D) SetTrainable(trainable bool) *LZeroPadding2D {
	l.trainable = trainable
	return l
}

func (l *LZeroPadding2D) SetLayerWeights(layerWeights interface{}) *LZeroPadding2D {
	l.layerWeights = layerWeights
	return l
}

func (l *LZeroPadding2D) GetShape() tf.Shape {
	return l.shape
}

func (l *LZeroPadding2D) GetDtype() DataType {
	return l.dtype
}

func (l *LZeroPadding2D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LZeroPadding2D) GetInputs() []Layer {
	return l.inputs
}

func (l *LZeroPadding2D) GetName() string {
	return l.name
}

func (l *LZeroPadding2D) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLZeroPadding2D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LZeroPadding2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLZeroPadding2D{
		ClassName: "ZeroPadding2D",
		Name:      l.name,
		Config: map[string]interface{}{
			"data_format": l.dataFormat,
			"dtype":       l.dtype.String(),
			"name":        l.name,
			"padding":     l.padding,
			"trainable":   l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LZeroPadding2D) GetCustomLayerDefinition() string {
	return ``
}

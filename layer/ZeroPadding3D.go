package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LZeroPadding3D struct {
	dataFormat interface{}
	dtype      DataType
	inputs     []Layer
	name       string
	padding    []interface{}
	shape      tf.Shape
	trainable  bool
}

func ZeroPadding3D() *LZeroPadding3D {
	return &LZeroPadding3D{
		dataFormat: nil,
		dtype:      Float32,
		name:       UniqueName("zero_padding3d"),
		padding:    []interface{}{1, 1, 1},
		trainable:  true,
	}
}

func (l *LZeroPadding3D) SetDataFormat(dataFormat interface{}) *LZeroPadding3D {
	l.dataFormat = dataFormat
	return l
}

func (l *LZeroPadding3D) SetDtype(dtype DataType) *LZeroPadding3D {
	l.dtype = dtype
	return l
}

func (l *LZeroPadding3D) SetName(name string) *LZeroPadding3D {
	l.name = name
	return l
}

func (l *LZeroPadding3D) SetPadding(padding []interface{}) *LZeroPadding3D {
	l.padding = padding
	return l
}

func (l *LZeroPadding3D) SetShape(shape tf.Shape) *LZeroPadding3D {
	l.shape = shape
	return l
}

func (l *LZeroPadding3D) SetTrainable(trainable bool) *LZeroPadding3D {
	l.trainable = trainable
	return l
}

func (l *LZeroPadding3D) GetShape() tf.Shape {
	return l.shape
}

func (l *LZeroPadding3D) GetDtype() DataType {
	return l.dtype
}

func (l *LZeroPadding3D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LZeroPadding3D) GetInputs() []Layer {
	return l.inputs
}

func (l *LZeroPadding3D) GetName() string {
	return l.name
}

type jsonConfigLZeroPadding3D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LZeroPadding3D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLZeroPadding3D{
		ClassName: "ZeroPadding3D",
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

func (l *LZeroPadding3D) GetCustomLayerDefinition() string {
	return ``
}

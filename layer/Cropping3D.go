package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LCropping3D struct {
	cropping   []interface{}
	dataFormat interface{}
	dtype      DataType
	inputs     []Layer
	name       string
	shape      tf.Shape
	trainable  bool
}

func Cropping3D() *LCropping3D {
	return &LCropping3D{
		cropping:   []interface{}{[]interface{}{1, 1}, []interface{}{1, 1}, []interface{}{1, 1}},
		dataFormat: nil,
		dtype:      Float32,
		name:       UniqueName("cropping3d"),
		trainable:  true,
	}
}

func (l *LCropping3D) SetCropping(cropping []interface{}) *LCropping3D {
	l.cropping = cropping
	return l
}

func (l *LCropping3D) SetDataFormat(dataFormat interface{}) *LCropping3D {
	l.dataFormat = dataFormat
	return l
}

func (l *LCropping3D) SetDtype(dtype DataType) *LCropping3D {
	l.dtype = dtype
	return l
}

func (l *LCropping3D) SetName(name string) *LCropping3D {
	l.name = name
	return l
}

func (l *LCropping3D) SetShape(shape tf.Shape) *LCropping3D {
	l.shape = shape
	return l
}

func (l *LCropping3D) SetTrainable(trainable bool) *LCropping3D {
	l.trainable = trainable
	return l
}

func (l *LCropping3D) GetShape() tf.Shape {
	return l.shape
}

func (l *LCropping3D) GetDtype() DataType {
	return l.dtype
}

func (l *LCropping3D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LCropping3D) GetInputs() []Layer {
	return l.inputs
}

func (l *LCropping3D) GetName() string {
	return l.name
}

type jsonConfigLCropping3D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LCropping3D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLCropping3D{
		ClassName: "Cropping3D",
		Name:      l.name,
		Config: map[string]interface{}{
			"cropping":    l.cropping,
			"data_format": l.dataFormat,
			"dtype":       l.dtype.String(),
			"name":        l.name,
			"trainable":   l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LCropping3D) GetCustomLayerDefinition() string {
	return ``
}

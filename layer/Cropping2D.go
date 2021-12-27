package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LCropping2D struct {
	cropping   []interface{}
	dataFormat interface{}
	dtype      DataType
	inputs     []Layer
	name       string
	shape      tf.Shape
	trainable  bool
}

func Cropping2D() *LCropping2D {
	return &LCropping2D{
		cropping:   []interface{}{[]interface{}{0, 0}, []interface{}{0, 0}},
		dataFormat: nil,
		dtype:      Float32,
		name:       UniqueName("cropping2d"),
		trainable:  true,
	}
}

func (l *LCropping2D) SetCropping(cropping []interface{}) *LCropping2D {
	l.cropping = cropping
	return l
}

func (l *LCropping2D) SetDataFormat(dataFormat interface{}) *LCropping2D {
	l.dataFormat = dataFormat
	return l
}

func (l *LCropping2D) SetDtype(dtype DataType) *LCropping2D {
	l.dtype = dtype
	return l
}

func (l *LCropping2D) SetName(name string) *LCropping2D {
	l.name = name
	return l
}

func (l *LCropping2D) SetShape(shape tf.Shape) *LCropping2D {
	l.shape = shape
	return l
}

func (l *LCropping2D) SetTrainable(trainable bool) *LCropping2D {
	l.trainable = trainable
	return l
}

func (l *LCropping2D) GetShape() tf.Shape {
	return l.shape
}

func (l *LCropping2D) GetDtype() DataType {
	return l.dtype
}

func (l *LCropping2D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LCropping2D) GetInputs() []Layer {
	return l.inputs
}

func (l *LCropping2D) GetName() string {
	return l.name
}

type jsonConfigLCropping2D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LCropping2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLCropping2D{
		ClassName: "Cropping2D",
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

func (l *LCropping2D) GetCustomLayerDefinition() string {
	return ``
}

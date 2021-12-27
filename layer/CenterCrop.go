package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LCenterCrop struct {
	dtype     DataType
	height    float64
	inputs    []Layer
	name      string
	shape     tf.Shape
	trainable bool
	width     float64
}

func CenterCrop(height float64, width float64) *LCenterCrop {
	return &LCenterCrop{
		dtype:     Float32,
		height:    height,
		name:      UniqueName("center_crop"),
		trainable: true,
		width:     width,
	}
}

func (l *LCenterCrop) SetDtype(dtype DataType) *LCenterCrop {
	l.dtype = dtype
	return l
}

func (l *LCenterCrop) SetName(name string) *LCenterCrop {
	l.name = name
	return l
}

func (l *LCenterCrop) SetShape(shape tf.Shape) *LCenterCrop {
	l.shape = shape
	return l
}

func (l *LCenterCrop) SetTrainable(trainable bool) *LCenterCrop {
	l.trainable = trainable
	return l
}

func (l *LCenterCrop) GetShape() tf.Shape {
	return l.shape
}

func (l *LCenterCrop) GetDtype() DataType {
	return l.dtype
}

func (l *LCenterCrop) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LCenterCrop) GetInputs() []Layer {
	return l.inputs
}

func (l *LCenterCrop) GetName() string {
	return l.name
}

type jsonConfigLCenterCrop struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LCenterCrop) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLCenterCrop{
		ClassName: "CenterCrop",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"height":    l.height,
			"name":      l.name,
			"trainable": l.trainable,
			"width":     l.width,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LCenterCrop) GetCustomLayerDefinition() string {
	return ``
}

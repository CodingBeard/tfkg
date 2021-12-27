package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LUpSampling3D struct {
	dataFormat interface{}
	dtype      DataType
	inputs     []Layer
	name       string
	shape      tf.Shape
	size       []interface{}
	trainable  bool
}

func UpSampling3D() *LUpSampling3D {
	return &LUpSampling3D{
		dataFormat: nil,
		dtype:      Float32,
		name:       UniqueName("up_sampling3d"),
		size:       []interface{}{2, 2, 2},
		trainable:  true,
	}
}

func (l *LUpSampling3D) SetDataFormat(dataFormat interface{}) *LUpSampling3D {
	l.dataFormat = dataFormat
	return l
}

func (l *LUpSampling3D) SetDtype(dtype DataType) *LUpSampling3D {
	l.dtype = dtype
	return l
}

func (l *LUpSampling3D) SetName(name string) *LUpSampling3D {
	l.name = name
	return l
}

func (l *LUpSampling3D) SetShape(shape tf.Shape) *LUpSampling3D {
	l.shape = shape
	return l
}

func (l *LUpSampling3D) SetSize(size []interface{}) *LUpSampling3D {
	l.size = size
	return l
}

func (l *LUpSampling3D) SetTrainable(trainable bool) *LUpSampling3D {
	l.trainable = trainable
	return l
}

func (l *LUpSampling3D) GetShape() tf.Shape {
	return l.shape
}

func (l *LUpSampling3D) GetDtype() DataType {
	return l.dtype
}

func (l *LUpSampling3D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LUpSampling3D) GetInputs() []Layer {
	return l.inputs
}

func (l *LUpSampling3D) GetName() string {
	return l.name
}

type jsonConfigLUpSampling3D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LUpSampling3D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLUpSampling3D{
		ClassName: "UpSampling3D",
		Name:      l.name,
		Config: map[string]interface{}{
			"data_format": l.dataFormat,
			"dtype":       l.dtype.String(),
			"name":        l.name,
			"size":        l.size,
			"trainable":   l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LUpSampling3D) GetCustomLayerDefinition() string {
	return ``
}

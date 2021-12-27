package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LUpSampling2D struct {
	dataFormat    interface{}
	dtype         DataType
	inputs        []Layer
	interpolation string
	name          string
	shape         tf.Shape
	size          []interface{}
	trainable     bool
}

func UpSampling2D() *LUpSampling2D {
	return &LUpSampling2D{
		dataFormat:    nil,
		dtype:         Float32,
		interpolation: "nearest",
		name:          UniqueName("up_sampling2d"),
		size:          []interface{}{2, 2},
		trainable:     true,
	}
}

func (l *LUpSampling2D) SetDataFormat(dataFormat interface{}) *LUpSampling2D {
	l.dataFormat = dataFormat
	return l
}

func (l *LUpSampling2D) SetDtype(dtype DataType) *LUpSampling2D {
	l.dtype = dtype
	return l
}

func (l *LUpSampling2D) SetInterpolation(interpolation string) *LUpSampling2D {
	l.interpolation = interpolation
	return l
}

func (l *LUpSampling2D) SetName(name string) *LUpSampling2D {
	l.name = name
	return l
}

func (l *LUpSampling2D) SetShape(shape tf.Shape) *LUpSampling2D {
	l.shape = shape
	return l
}

func (l *LUpSampling2D) SetSize(size []interface{}) *LUpSampling2D {
	l.size = size
	return l
}

func (l *LUpSampling2D) SetTrainable(trainable bool) *LUpSampling2D {
	l.trainable = trainable
	return l
}

func (l *LUpSampling2D) GetShape() tf.Shape {
	return l.shape
}

func (l *LUpSampling2D) GetDtype() DataType {
	return l.dtype
}

func (l *LUpSampling2D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LUpSampling2D) GetInputs() []Layer {
	return l.inputs
}

func (l *LUpSampling2D) GetName() string {
	return l.name
}

type jsonConfigLUpSampling2D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LUpSampling2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLUpSampling2D{
		ClassName: "UpSampling2D",
		Name:      l.name,
		Config: map[string]interface{}{
			"data_format":   l.dataFormat,
			"dtype":         l.dtype.String(),
			"interpolation": l.interpolation,
			"name":          l.name,
			"size":          l.size,
			"trainable":     l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LUpSampling2D) GetCustomLayerDefinition() string {
	return ``
}

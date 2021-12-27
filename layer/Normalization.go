package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LNormalization struct {
	axis      float64
	dtype     DataType
	inputs    []Layer
	mean      interface{}
	name      string
	shape     tf.Shape
	trainable bool
	variance  interface{}
}

func Normalization() *LNormalization {
	return &LNormalization{
		axis:      -1,
		dtype:     Float32,
		mean:      nil,
		name:      UniqueName("normalization"),
		trainable: true,
		variance:  nil,
	}
}

func (l *LNormalization) SetAxis(axis float64) *LNormalization {
	l.axis = axis
	return l
}

func (l *LNormalization) SetDtype(dtype DataType) *LNormalization {
	l.dtype = dtype
	return l
}

func (l *LNormalization) SetMean(mean interface{}) *LNormalization {
	l.mean = mean
	return l
}

func (l *LNormalization) SetName(name string) *LNormalization {
	l.name = name
	return l
}

func (l *LNormalization) SetShape(shape tf.Shape) *LNormalization {
	l.shape = shape
	return l
}

func (l *LNormalization) SetTrainable(trainable bool) *LNormalization {
	l.trainable = trainable
	return l
}

func (l *LNormalization) SetVariance(variance interface{}) *LNormalization {
	l.variance = variance
	return l
}

func (l *LNormalization) GetShape() tf.Shape {
	return l.shape
}

func (l *LNormalization) GetDtype() DataType {
	return l.dtype
}

func (l *LNormalization) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LNormalization) GetInputs() []Layer {
	return l.inputs
}

func (l *LNormalization) GetName() string {
	return l.name
}

type jsonConfigLNormalization struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LNormalization) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLNormalization{
		ClassName: "Normalization",
		Name:      l.name,
		Config: map[string]interface{}{
			"axis":      l.axis,
			"dtype":     l.dtype.String(),
			"mean":      l.mean,
			"name":      l.name,
			"trainable": l.trainable,
			"variance":  l.variance,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LNormalization) GetCustomLayerDefinition() string {
	return ``
}

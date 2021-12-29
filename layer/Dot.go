package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LDot struct {
	axes         float64
	dtype        DataType
	inputs       []Layer
	name         string
	normalize    bool
	shape        tf.Shape
	trainable    bool
	layerWeights interface{}
}

func Dot(axes float64) *LDot {
	return &LDot{
		axes:      axes,
		dtype:     Float32,
		name:      UniqueName("dot"),
		normalize: false,
		trainable: true,
	}
}

func (l *LDot) SetDtype(dtype DataType) *LDot {
	l.dtype = dtype
	return l
}

func (l *LDot) SetName(name string) *LDot {
	l.name = name
	return l
}

func (l *LDot) SetNormalize(normalize bool) *LDot {
	l.normalize = normalize
	return l
}

func (l *LDot) SetShape(shape tf.Shape) *LDot {
	l.shape = shape
	return l
}

func (l *LDot) SetTrainable(trainable bool) *LDot {
	l.trainable = trainable
	return l
}

func (l *LDot) SetLayerWeights(layerWeights interface{}) *LDot {
	l.layerWeights = layerWeights
	return l
}

func (l *LDot) GetShape() tf.Shape {
	return l.shape
}

func (l *LDot) GetDtype() DataType {
	return l.dtype
}

func (l *LDot) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LDot) GetInputs() []Layer {
	return l.inputs
}

func (l *LDot) GetName() string {
	return l.name
}

func (l *LDot) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLDot struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LDot) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLDot{
		ClassName: "Dot",
		Name:      l.name,
		Config: map[string]interface{}{
			"axes":      l.axes,
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"normalize": l.normalize,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LDot) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LConcatenate struct {
	axis         float64
	dtype        DataType
	inputs       []Layer
	name         string
	shape        tf.Shape
	trainable    bool
	layerWeights []*tf.Tensor
}

func Concatenate() *LConcatenate {
	return &LConcatenate{
		axis:      -1,
		dtype:     Float32,
		name:      UniqueName("concatenate"),
		trainable: true,
	}
}

func (l *LConcatenate) SetAxis(axis float64) *LConcatenate {
	l.axis = axis
	return l
}

func (l *LConcatenate) SetDtype(dtype DataType) *LConcatenate {
	l.dtype = dtype
	return l
}

func (l *LConcatenate) SetName(name string) *LConcatenate {
	l.name = name
	return l
}

func (l *LConcatenate) SetShape(shape tf.Shape) *LConcatenate {
	l.shape = shape
	return l
}

func (l *LConcatenate) SetTrainable(trainable bool) *LConcatenate {
	l.trainable = trainable
	return l
}

func (l *LConcatenate) SetLayerWeights(layerWeights []*tf.Tensor) *LConcatenate {
	l.layerWeights = layerWeights
	return l
}

func (l *LConcatenate) GetShape() tf.Shape {
	return l.shape
}

func (l *LConcatenate) GetDtype() DataType {
	return l.dtype
}

func (l *LConcatenate) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LConcatenate) GetInputs() []Layer {
	return l.inputs
}

func (l *LConcatenate) GetName() string {
	return l.name
}

func (l *LConcatenate) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLConcatenate struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LConcatenate) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLConcatenate{
		ClassName: "Concatenate",
		Name:      l.name,
		Config: map[string]interface{}{
			"axis":      l.axis,
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LConcatenate) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LCategoryCrossing struct {
	depth        interface{}
	dtype        DataType
	inputs       []Layer
	name         string
	separator    string
	shape        tf.Shape
	trainable    bool
	layerWeights interface{}
}

func CategoryCrossing() *LCategoryCrossing {
	return &LCategoryCrossing{
		depth:     nil,
		dtype:     Float32,
		name:      UniqueName("nil"),
		separator: "_X_",
		trainable: true,
	}
}

func (l *LCategoryCrossing) SetDepth(depth interface{}) *LCategoryCrossing {
	l.depth = depth
	return l
}

func (l *LCategoryCrossing) SetDtype(dtype DataType) *LCategoryCrossing {
	l.dtype = dtype
	return l
}

func (l *LCategoryCrossing) SetName(name string) *LCategoryCrossing {
	l.name = name
	return l
}

func (l *LCategoryCrossing) SetSeparator(separator string) *LCategoryCrossing {
	l.separator = separator
	return l
}

func (l *LCategoryCrossing) SetShape(shape tf.Shape) *LCategoryCrossing {
	l.shape = shape
	return l
}

func (l *LCategoryCrossing) SetTrainable(trainable bool) *LCategoryCrossing {
	l.trainable = trainable
	return l
}

func (l *LCategoryCrossing) SetLayerWeights(layerWeights interface{}) *LCategoryCrossing {
	l.layerWeights = layerWeights
	return l
}

func (l *LCategoryCrossing) GetShape() tf.Shape {
	return l.shape
}

func (l *LCategoryCrossing) GetDtype() DataType {
	return l.dtype
}

func (l *LCategoryCrossing) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LCategoryCrossing) GetInputs() []Layer {
	return l.inputs
}

func (l *LCategoryCrossing) GetName() string {
	return l.name
}

func (l *LCategoryCrossing) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLCategoryCrossing struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LCategoryCrossing) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLCategoryCrossing{
		ClassName: "CategoryCrossing",
		Name:      l.name,
		Config: map[string]interface{}{
			"depth":     l.depth,
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"separator": l.separator,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LCategoryCrossing) GetCustomLayerDefinition() string {
	return ``
}

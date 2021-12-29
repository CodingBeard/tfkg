package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LMaximum struct {
	dtype        DataType
	inputs       []Layer
	name         string
	shape        tf.Shape
	trainable    bool
	layerWeights interface{}
}

func Maximum() *LMaximum {
	return &LMaximum{
		dtype:     Float32,
		name:      UniqueName("maximum"),
		trainable: true,
	}
}

func (l *LMaximum) SetDtype(dtype DataType) *LMaximum {
	l.dtype = dtype
	return l
}

func (l *LMaximum) SetName(name string) *LMaximum {
	l.name = name
	return l
}

func (l *LMaximum) SetShape(shape tf.Shape) *LMaximum {
	l.shape = shape
	return l
}

func (l *LMaximum) SetTrainable(trainable bool) *LMaximum {
	l.trainable = trainable
	return l
}

func (l *LMaximum) SetLayerWeights(layerWeights interface{}) *LMaximum {
	l.layerWeights = layerWeights
	return l
}

func (l *LMaximum) GetShape() tf.Shape {
	return l.shape
}

func (l *LMaximum) GetDtype() DataType {
	return l.dtype
}

func (l *LMaximum) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LMaximum) GetInputs() []Layer {
	return l.inputs
}

func (l *LMaximum) GetName() string {
	return l.name
}

func (l *LMaximum) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLMaximum struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LMaximum) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLMaximum{
		ClassName: "Maximum",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LMaximum) GetCustomLayerDefinition() string {
	return ``
}

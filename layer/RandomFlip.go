package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LRandomFlip struct {
	dtype        DataType
	inputs       []Layer
	mode         string
	name         string
	seed         interface{}
	shape        tf.Shape
	trainable    bool
	layerWeights interface{}
}

func RandomFlip() *LRandomFlip {
	return &LRandomFlip{
		dtype:     Float32,
		mode:      "horizontal_and_vertical",
		name:      UniqueName("random_flip"),
		seed:      nil,
		trainable: true,
	}
}

func (l *LRandomFlip) SetDtype(dtype DataType) *LRandomFlip {
	l.dtype = dtype
	return l
}

func (l *LRandomFlip) SetMode(mode string) *LRandomFlip {
	l.mode = mode
	return l
}

func (l *LRandomFlip) SetName(name string) *LRandomFlip {
	l.name = name
	return l
}

func (l *LRandomFlip) SetSeed(seed interface{}) *LRandomFlip {
	l.seed = seed
	return l
}

func (l *LRandomFlip) SetShape(shape tf.Shape) *LRandomFlip {
	l.shape = shape
	return l
}

func (l *LRandomFlip) SetTrainable(trainable bool) *LRandomFlip {
	l.trainable = trainable
	return l
}

func (l *LRandomFlip) SetLayerWeights(layerWeights interface{}) *LRandomFlip {
	l.layerWeights = layerWeights
	return l
}

func (l *LRandomFlip) GetShape() tf.Shape {
	return l.shape
}

func (l *LRandomFlip) GetDtype() DataType {
	return l.dtype
}

func (l *LRandomFlip) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LRandomFlip) GetInputs() []Layer {
	return l.inputs
}

func (l *LRandomFlip) GetName() string {
	return l.name
}

func (l *LRandomFlip) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLRandomFlip struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LRandomFlip) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLRandomFlip{
		ClassName: "RandomFlip",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"mode":      l.mode,
			"name":      l.name,
			"seed":      l.seed,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LRandomFlip) GetCustomLayerDefinition() string {
	return ``
}

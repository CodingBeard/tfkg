package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LRandomCrop struct {
	dtype     DataType
	height    float64
	inputs    []Layer
	name      string
	seed      interface{}
	shape     tf.Shape
	trainable bool
	width     float64
}

func RandomCrop(height float64, width float64) *LRandomCrop {
	return &LRandomCrop{
		dtype:     Float32,
		height:    height,
		name:      UniqueName("random_crop"),
		seed:      nil,
		trainable: true,
		width:     width,
	}
}

func (l *LRandomCrop) SetDtype(dtype DataType) *LRandomCrop {
	l.dtype = dtype
	return l
}

func (l *LRandomCrop) SetName(name string) *LRandomCrop {
	l.name = name
	return l
}

func (l *LRandomCrop) SetSeed(seed interface{}) *LRandomCrop {
	l.seed = seed
	return l
}

func (l *LRandomCrop) SetShape(shape tf.Shape) *LRandomCrop {
	l.shape = shape
	return l
}

func (l *LRandomCrop) SetTrainable(trainable bool) *LRandomCrop {
	l.trainable = trainable
	return l
}

func (l *LRandomCrop) GetShape() tf.Shape {
	return l.shape
}

func (l *LRandomCrop) GetDtype() DataType {
	return l.dtype
}

func (l *LRandomCrop) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LRandomCrop) GetInputs() []Layer {
	return l.inputs
}

func (l *LRandomCrop) GetName() string {
	return l.name
}

type jsonConfigLRandomCrop struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LRandomCrop) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLRandomCrop{
		ClassName: "RandomCrop",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"height":    l.height,
			"name":      l.name,
			"seed":      l.seed,
			"trainable": l.trainable,
			"width":     l.width,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LRandomCrop) GetCustomLayerDefinition() string {
	return ``
}

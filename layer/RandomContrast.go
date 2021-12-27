package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LRandomContrast struct {
	dtype     DataType
	factor    float64
	inputs    []Layer
	name      string
	seed      interface{}
	shape     tf.Shape
	trainable bool
}

func RandomContrast(factor float64) *LRandomContrast {
	return &LRandomContrast{
		dtype:     Float32,
		factor:    factor,
		name:      UniqueName("random_contrast"),
		seed:      nil,
		trainable: true,
	}
}

func (l *LRandomContrast) SetDtype(dtype DataType) *LRandomContrast {
	l.dtype = dtype
	return l
}

func (l *LRandomContrast) SetName(name string) *LRandomContrast {
	l.name = name
	return l
}

func (l *LRandomContrast) SetSeed(seed interface{}) *LRandomContrast {
	l.seed = seed
	return l
}

func (l *LRandomContrast) SetShape(shape tf.Shape) *LRandomContrast {
	l.shape = shape
	return l
}

func (l *LRandomContrast) SetTrainable(trainable bool) *LRandomContrast {
	l.trainable = trainable
	return l
}

func (l *LRandomContrast) GetShape() tf.Shape {
	return l.shape
}

func (l *LRandomContrast) GetDtype() DataType {
	return l.dtype
}

func (l *LRandomContrast) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LRandomContrast) GetInputs() []Layer {
	return l.inputs
}

func (l *LRandomContrast) GetName() string {
	return l.name
}

type jsonConfigLRandomContrast struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LRandomContrast) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLRandomContrast{
		ClassName: "RandomContrast",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"factor":    l.factor,
			"name":      l.name,
			"seed":      l.seed,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LRandomContrast) GetCustomLayerDefinition() string {
	return ``
}

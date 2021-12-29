package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LRandomHeight struct {
	dtype         DataType
	factor        float64
	inputs        []Layer
	interpolation string
	name          string
	seed          interface{}
	shape         tf.Shape
	trainable     bool
	layerWeights  interface{}
}

func RandomHeight(factor float64) *LRandomHeight {
	return &LRandomHeight{
		dtype:         Float32,
		factor:        factor,
		interpolation: "bilinear",
		name:          UniqueName("random_height"),
		seed:          nil,
		trainable:     true,
	}
}

func (l *LRandomHeight) SetDtype(dtype DataType) *LRandomHeight {
	l.dtype = dtype
	return l
}

func (l *LRandomHeight) SetInterpolation(interpolation string) *LRandomHeight {
	l.interpolation = interpolation
	return l
}

func (l *LRandomHeight) SetName(name string) *LRandomHeight {
	l.name = name
	return l
}

func (l *LRandomHeight) SetSeed(seed interface{}) *LRandomHeight {
	l.seed = seed
	return l
}

func (l *LRandomHeight) SetShape(shape tf.Shape) *LRandomHeight {
	l.shape = shape
	return l
}

func (l *LRandomHeight) SetTrainable(trainable bool) *LRandomHeight {
	l.trainable = trainable
	return l
}

func (l *LRandomHeight) SetLayerWeights(layerWeights interface{}) *LRandomHeight {
	l.layerWeights = layerWeights
	return l
}

func (l *LRandomHeight) GetShape() tf.Shape {
	return l.shape
}

func (l *LRandomHeight) GetDtype() DataType {
	return l.dtype
}

func (l *LRandomHeight) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LRandomHeight) GetInputs() []Layer {
	return l.inputs
}

func (l *LRandomHeight) GetName() string {
	return l.name
}

func (l *LRandomHeight) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLRandomHeight struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LRandomHeight) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLRandomHeight{
		ClassName: "RandomHeight",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":         l.dtype.String(),
			"factor":        l.factor,
			"interpolation": l.interpolation,
			"name":          l.name,
			"seed":          l.seed,
			"trainable":     l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LRandomHeight) GetCustomLayerDefinition() string {
	return ``
}

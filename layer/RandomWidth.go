package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LRandomWidth struct {
	dtype         DataType
	factor        float64
	inputs        []Layer
	interpolation string
	name          string
	seed          interface{}
	shape         tf.Shape
	trainable     bool
	layerWeights  []*tf.Tensor
}

func RandomWidth(factor float64) *LRandomWidth {
	return &LRandomWidth{
		dtype:         Float32,
		factor:        factor,
		interpolation: "bilinear",
		name:          UniqueName("random_width"),
		seed:          nil,
		trainable:     true,
	}
}

func (l *LRandomWidth) SetDtype(dtype DataType) *LRandomWidth {
	l.dtype = dtype
	return l
}

func (l *LRandomWidth) SetInterpolation(interpolation string) *LRandomWidth {
	l.interpolation = interpolation
	return l
}

func (l *LRandomWidth) SetName(name string) *LRandomWidth {
	l.name = name
	return l
}

func (l *LRandomWidth) SetSeed(seed interface{}) *LRandomWidth {
	l.seed = seed
	return l
}

func (l *LRandomWidth) SetShape(shape tf.Shape) *LRandomWidth {
	l.shape = shape
	return l
}

func (l *LRandomWidth) SetTrainable(trainable bool) *LRandomWidth {
	l.trainable = trainable
	return l
}

func (l *LRandomWidth) SetLayerWeights(layerWeights []*tf.Tensor) *LRandomWidth {
	l.layerWeights = layerWeights
	return l
}

func (l *LRandomWidth) GetShape() tf.Shape {
	return l.shape
}

func (l *LRandomWidth) GetDtype() DataType {
	return l.dtype
}

func (l *LRandomWidth) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LRandomWidth) GetInputs() []Layer {
	return l.inputs
}

func (l *LRandomWidth) GetName() string {
	return l.name
}

func (l *LRandomWidth) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLRandomWidth struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LRandomWidth) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLRandomWidth{
		ClassName: "RandomWidth",
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

func (l *LRandomWidth) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LRandomTranslation struct {
	dtype         DataType
	fillMode      string
	fillValue     float64
	heightFactor  float64
	inputs        []Layer
	interpolation string
	name          string
	seed          interface{}
	shape         tf.Shape
	trainable     bool
	widthFactor   float64
	layerWeights  []*tf.Tensor
}

func RandomTranslation(heightFactor float64, widthFactor float64) *LRandomTranslation {
	return &LRandomTranslation{
		dtype:         Float32,
		fillMode:      "reflect",
		fillValue:     0,
		heightFactor:  heightFactor,
		interpolation: "bilinear",
		name:          UniqueName("random_translation"),
		seed:          nil,
		trainable:     true,
		widthFactor:   widthFactor,
	}
}

func (l *LRandomTranslation) SetDtype(dtype DataType) *LRandomTranslation {
	l.dtype = dtype
	return l
}

func (l *LRandomTranslation) SetFillMode(fillMode string) *LRandomTranslation {
	l.fillMode = fillMode
	return l
}

func (l *LRandomTranslation) SetFillValue(fillValue float64) *LRandomTranslation {
	l.fillValue = fillValue
	return l
}

func (l *LRandomTranslation) SetInterpolation(interpolation string) *LRandomTranslation {
	l.interpolation = interpolation
	return l
}

func (l *LRandomTranslation) SetName(name string) *LRandomTranslation {
	l.name = name
	return l
}

func (l *LRandomTranslation) SetSeed(seed interface{}) *LRandomTranslation {
	l.seed = seed
	return l
}

func (l *LRandomTranslation) SetShape(shape tf.Shape) *LRandomTranslation {
	l.shape = shape
	return l
}

func (l *LRandomTranslation) SetTrainable(trainable bool) *LRandomTranslation {
	l.trainable = trainable
	return l
}

func (l *LRandomTranslation) SetLayerWeights(layerWeights []*tf.Tensor) *LRandomTranslation {
	l.layerWeights = layerWeights
	return l
}

func (l *LRandomTranslation) GetShape() tf.Shape {
	return l.shape
}

func (l *LRandomTranslation) GetDtype() DataType {
	return l.dtype
}

func (l *LRandomTranslation) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LRandomTranslation) GetInputs() []Layer {
	return l.inputs
}

func (l *LRandomTranslation) GetName() string {
	return l.name
}

func (l *LRandomTranslation) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLRandomTranslation struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LRandomTranslation) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLRandomTranslation{
		ClassName: "RandomTranslation",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":         l.dtype.String(),
			"fill_mode":     l.fillMode,
			"fill_value":    l.fillValue,
			"height_factor": l.heightFactor,
			"interpolation": l.interpolation,
			"name":          l.name,
			"seed":          l.seed,
			"trainable":     l.trainable,
			"width_factor":  l.widthFactor,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LRandomTranslation) GetCustomLayerDefinition() string {
	return ``
}

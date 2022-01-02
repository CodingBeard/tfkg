package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LRandomZoom struct {
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
	widthFactor   interface{}
	layerWeights  []*tf.Tensor
}

func RandomZoom(heightFactor float64) *LRandomZoom {
	return &LRandomZoom{
		dtype:         Float32,
		fillMode:      "reflect",
		fillValue:     0,
		heightFactor:  heightFactor,
		interpolation: "bilinear",
		name:          UniqueName("random_zoom"),
		seed:          nil,
		trainable:     true,
		widthFactor:   nil,
	}
}

func (l *LRandomZoom) SetDtype(dtype DataType) *LRandomZoom {
	l.dtype = dtype
	return l
}

func (l *LRandomZoom) SetFillMode(fillMode string) *LRandomZoom {
	l.fillMode = fillMode
	return l
}

func (l *LRandomZoom) SetFillValue(fillValue float64) *LRandomZoom {
	l.fillValue = fillValue
	return l
}

func (l *LRandomZoom) SetInterpolation(interpolation string) *LRandomZoom {
	l.interpolation = interpolation
	return l
}

func (l *LRandomZoom) SetName(name string) *LRandomZoom {
	l.name = name
	return l
}

func (l *LRandomZoom) SetSeed(seed interface{}) *LRandomZoom {
	l.seed = seed
	return l
}

func (l *LRandomZoom) SetShape(shape tf.Shape) *LRandomZoom {
	l.shape = shape
	return l
}

func (l *LRandomZoom) SetTrainable(trainable bool) *LRandomZoom {
	l.trainable = trainable
	return l
}

func (l *LRandomZoom) SetWidthFactor(widthFactor interface{}) *LRandomZoom {
	l.widthFactor = widthFactor
	return l
}

func (l *LRandomZoom) SetLayerWeights(layerWeights []*tf.Tensor) *LRandomZoom {
	l.layerWeights = layerWeights
	return l
}

func (l *LRandomZoom) GetShape() tf.Shape {
	return l.shape
}

func (l *LRandomZoom) GetDtype() DataType {
	return l.dtype
}

func (l *LRandomZoom) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LRandomZoom) GetInputs() []Layer {
	return l.inputs
}

func (l *LRandomZoom) GetName() string {
	return l.name
}

func (l *LRandomZoom) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLRandomZoom struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LRandomZoom) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLRandomZoom{
		ClassName: "RandomZoom",
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

func (l *LRandomZoom) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LRandomRotation struct {
	dtype         DataType
	factor        float64
	fillMode      string
	fillValue     float64
	inputs        []Layer
	interpolation string
	name          string
	seed          interface{}
	shape         tf.Shape
	trainable     bool
}

func RandomRotation(factor float64) *LRandomRotation {
	return &LRandomRotation{
		dtype:         Float32,
		factor:        factor,
		fillMode:      "reflect",
		fillValue:     0,
		interpolation: "bilinear",
		name:          UniqueName("random_rotation"),
		seed:          nil,
		trainable:     true,
	}
}

func (l *LRandomRotation) SetDtype(dtype DataType) *LRandomRotation {
	l.dtype = dtype
	return l
}

func (l *LRandomRotation) SetFillMode(fillMode string) *LRandomRotation {
	l.fillMode = fillMode
	return l
}

func (l *LRandomRotation) SetFillValue(fillValue float64) *LRandomRotation {
	l.fillValue = fillValue
	return l
}

func (l *LRandomRotation) SetInterpolation(interpolation string) *LRandomRotation {
	l.interpolation = interpolation
	return l
}

func (l *LRandomRotation) SetName(name string) *LRandomRotation {
	l.name = name
	return l
}

func (l *LRandomRotation) SetSeed(seed interface{}) *LRandomRotation {
	l.seed = seed
	return l
}

func (l *LRandomRotation) SetShape(shape tf.Shape) *LRandomRotation {
	l.shape = shape
	return l
}

func (l *LRandomRotation) SetTrainable(trainable bool) *LRandomRotation {
	l.trainable = trainable
	return l
}

func (l *LRandomRotation) GetShape() tf.Shape {
	return l.shape
}

func (l *LRandomRotation) GetDtype() DataType {
	return l.dtype
}

func (l *LRandomRotation) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LRandomRotation) GetInputs() []Layer {
	return l.inputs
}

func (l *LRandomRotation) GetName() string {
	return l.name
}

type jsonConfigLRandomRotation struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LRandomRotation) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLRandomRotation{
		ClassName: "RandomRotation",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":         l.dtype.String(),
			"factor":        l.factor,
			"fill_mode":     l.fillMode,
			"fill_value":    l.fillValue,
			"interpolation": l.interpolation,
			"name":          l.name,
			"seed":          l.seed,
			"trainable":     l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LRandomRotation) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LSpatialDropout1D struct {
	dtype      DataType
	inputs     []Layer
	name       string
	noiseShape interface{}
	rate       float64
	seed       interface{}
	shape      tf.Shape
	trainable  bool
}

func SpatialDropout1D(rate float64) *LSpatialDropout1D {
	return &LSpatialDropout1D{
		dtype:      Float32,
		name:       UniqueName("spatial_dropout1d"),
		noiseShape: nil,
		rate:       rate,
		seed:       nil,
		trainable:  true,
	}
}

func (l *LSpatialDropout1D) SetDtype(dtype DataType) *LSpatialDropout1D {
	l.dtype = dtype
	return l
}

func (l *LSpatialDropout1D) SetName(name string) *LSpatialDropout1D {
	l.name = name
	return l
}

func (l *LSpatialDropout1D) SetNoiseShape(noiseShape interface{}) *LSpatialDropout1D {
	l.noiseShape = noiseShape
	return l
}

func (l *LSpatialDropout1D) SetSeed(seed interface{}) *LSpatialDropout1D {
	l.seed = seed
	return l
}

func (l *LSpatialDropout1D) SetShape(shape tf.Shape) *LSpatialDropout1D {
	l.shape = shape
	return l
}

func (l *LSpatialDropout1D) SetTrainable(trainable bool) *LSpatialDropout1D {
	l.trainable = trainable
	return l
}

func (l *LSpatialDropout1D) GetShape() tf.Shape {
	return l.shape
}

func (l *LSpatialDropout1D) GetDtype() DataType {
	return l.dtype
}

func (l *LSpatialDropout1D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LSpatialDropout1D) GetInputs() []Layer {
	return l.inputs
}

func (l *LSpatialDropout1D) GetName() string {
	return l.name
}

type jsonConfigLSpatialDropout1D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LSpatialDropout1D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLSpatialDropout1D{
		ClassName: "SpatialDropout1D",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":       l.dtype.String(),
			"name":        l.name,
			"noise_shape": l.noiseShape,
			"rate":        l.rate,
			"seed":        l.seed,
			"trainable":   l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LSpatialDropout1D) GetCustomLayerDefinition() string {
	return ``
}

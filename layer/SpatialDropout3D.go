package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LSpatialDropout3D struct {
	dataFormat interface{}
	dtype      DataType
	inputs     []Layer
	name       string
	noiseShape interface{}
	rate       float64
	seed       interface{}
	shape      tf.Shape
	trainable  bool
}

func SpatialDropout3D(rate float64) *LSpatialDropout3D {
	return &LSpatialDropout3D{
		dataFormat: nil,
		dtype:      Float32,
		name:       UniqueName("spatial_dropout3d"),
		noiseShape: nil,
		rate:       rate,
		seed:       nil,
		trainable:  true,
	}
}

func (l *LSpatialDropout3D) SetDataFormat(dataFormat interface{}) *LSpatialDropout3D {
	l.dataFormat = dataFormat
	return l
}

func (l *LSpatialDropout3D) SetDtype(dtype DataType) *LSpatialDropout3D {
	l.dtype = dtype
	return l
}

func (l *LSpatialDropout3D) SetName(name string) *LSpatialDropout3D {
	l.name = name
	return l
}

func (l *LSpatialDropout3D) SetNoiseShape(noiseShape interface{}) *LSpatialDropout3D {
	l.noiseShape = noiseShape
	return l
}

func (l *LSpatialDropout3D) SetSeed(seed interface{}) *LSpatialDropout3D {
	l.seed = seed
	return l
}

func (l *LSpatialDropout3D) SetShape(shape tf.Shape) *LSpatialDropout3D {
	l.shape = shape
	return l
}

func (l *LSpatialDropout3D) SetTrainable(trainable bool) *LSpatialDropout3D {
	l.trainable = trainable
	return l
}

func (l *LSpatialDropout3D) GetShape() tf.Shape {
	return l.shape
}

func (l *LSpatialDropout3D) GetDtype() DataType {
	return l.dtype
}

func (l *LSpatialDropout3D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LSpatialDropout3D) GetInputs() []Layer {
	return l.inputs
}

func (l *LSpatialDropout3D) GetName() string {
	return l.name
}

type jsonConfigLSpatialDropout3D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LSpatialDropout3D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLSpatialDropout3D{
		ClassName: "SpatialDropout3D",
		Name:      l.name,
		Config: map[string]interface{}{
			"data_format": l.dataFormat,
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

func (l *LSpatialDropout3D) GetCustomLayerDefinition() string {
	return ``
}

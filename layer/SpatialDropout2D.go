package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LSpatialDropout2D struct {
	dataFormat   interface{}
	dtype        DataType
	inputs       []Layer
	name         string
	noiseShape   interface{}
	rate         float64
	seed         interface{}
	shape        tf.Shape
	trainable    bool
	layerWeights []*tf.Tensor
}

func SpatialDropout2D(rate float64) *LSpatialDropout2D {
	return &LSpatialDropout2D{
		dataFormat: nil,
		dtype:      Float32,
		name:       UniqueName("spatial_dropout2d"),
		noiseShape: nil,
		rate:       rate,
		seed:       nil,
		trainable:  true,
	}
}

func (l *LSpatialDropout2D) SetDataFormat(dataFormat interface{}) *LSpatialDropout2D {
	l.dataFormat = dataFormat
	return l
}

func (l *LSpatialDropout2D) SetDtype(dtype DataType) *LSpatialDropout2D {
	l.dtype = dtype
	return l
}

func (l *LSpatialDropout2D) SetName(name string) *LSpatialDropout2D {
	l.name = name
	return l
}

func (l *LSpatialDropout2D) SetNoiseShape(noiseShape interface{}) *LSpatialDropout2D {
	l.noiseShape = noiseShape
	return l
}

func (l *LSpatialDropout2D) SetSeed(seed interface{}) *LSpatialDropout2D {
	l.seed = seed
	return l
}

func (l *LSpatialDropout2D) SetShape(shape tf.Shape) *LSpatialDropout2D {
	l.shape = shape
	return l
}

func (l *LSpatialDropout2D) SetTrainable(trainable bool) *LSpatialDropout2D {
	l.trainable = trainable
	return l
}

func (l *LSpatialDropout2D) SetLayerWeights(layerWeights []*tf.Tensor) *LSpatialDropout2D {
	l.layerWeights = layerWeights
	return l
}

func (l *LSpatialDropout2D) GetShape() tf.Shape {
	return l.shape
}

func (l *LSpatialDropout2D) GetDtype() DataType {
	return l.dtype
}

func (l *LSpatialDropout2D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LSpatialDropout2D) GetInputs() []Layer {
	return l.inputs
}

func (l *LSpatialDropout2D) GetName() string {
	return l.name
}

func (l *LSpatialDropout2D) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLSpatialDropout2D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LSpatialDropout2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLSpatialDropout2D{
		ClassName: "SpatialDropout2D",
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

func (l *LSpatialDropout2D) GetCustomLayerDefinition() string {
	return ``
}

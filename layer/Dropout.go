package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LDropout struct {
	dtype        DataType
	inputs       []Layer
	name         string
	noiseShape   interface{}
	rate         float64
	seed         interface{}
	shape        tf.Shape
	trainable    bool
	layerWeights interface{}
}

func Dropout(rate float64) *LDropout {
	return &LDropout{
		dtype:      Float32,
		name:       UniqueName("dropout"),
		noiseShape: nil,
		rate:       rate,
		seed:       nil,
		trainable:  true,
	}
}

func (l *LDropout) SetDtype(dtype DataType) *LDropout {
	l.dtype = dtype
	return l
}

func (l *LDropout) SetName(name string) *LDropout {
	l.name = name
	return l
}

func (l *LDropout) SetNoiseShape(noiseShape interface{}) *LDropout {
	l.noiseShape = noiseShape
	return l
}

func (l *LDropout) SetSeed(seed interface{}) *LDropout {
	l.seed = seed
	return l
}

func (l *LDropout) SetShape(shape tf.Shape) *LDropout {
	l.shape = shape
	return l
}

func (l *LDropout) SetTrainable(trainable bool) *LDropout {
	l.trainable = trainable
	return l
}

func (l *LDropout) SetLayerWeights(layerWeights interface{}) *LDropout {
	l.layerWeights = layerWeights
	return l
}

func (l *LDropout) GetShape() tf.Shape {
	return l.shape
}

func (l *LDropout) GetDtype() DataType {
	return l.dtype
}

func (l *LDropout) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LDropout) GetInputs() []Layer {
	return l.inputs
}

func (l *LDropout) GetName() string {
	return l.name
}

func (l *LDropout) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLDropout struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LDropout) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLDropout{
		ClassName: "Dropout",
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

func (l *LDropout) GetCustomLayerDefinition() string {
	return ``
}

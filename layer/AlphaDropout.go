package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LAlphaDropout struct {
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

func AlphaDropout(rate float64) *LAlphaDropout {
	return &LAlphaDropout{
		dtype:      Float32,
		name:       UniqueName("alpha_dropout"),
		noiseShape: nil,
		rate:       rate,
		seed:       nil,
		trainable:  true,
	}
}

func (l *LAlphaDropout) SetDtype(dtype DataType) *LAlphaDropout {
	l.dtype = dtype
	return l
}

func (l *LAlphaDropout) SetName(name string) *LAlphaDropout {
	l.name = name
	return l
}

func (l *LAlphaDropout) SetNoiseShape(noiseShape interface{}) *LAlphaDropout {
	l.noiseShape = noiseShape
	return l
}

func (l *LAlphaDropout) SetSeed(seed interface{}) *LAlphaDropout {
	l.seed = seed
	return l
}

func (l *LAlphaDropout) SetShape(shape tf.Shape) *LAlphaDropout {
	l.shape = shape
	return l
}

func (l *LAlphaDropout) SetTrainable(trainable bool) *LAlphaDropout {
	l.trainable = trainable
	return l
}

func (l *LAlphaDropout) SetLayerWeights(layerWeights []*tf.Tensor) *LAlphaDropout {
	l.layerWeights = layerWeights
	return l
}

func (l *LAlphaDropout) GetShape() tf.Shape {
	return l.shape
}

func (l *LAlphaDropout) GetDtype() DataType {
	return l.dtype
}

func (l *LAlphaDropout) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LAlphaDropout) GetInputs() []Layer {
	return l.inputs
}

func (l *LAlphaDropout) GetName() string {
	return l.name
}

func (l *LAlphaDropout) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLAlphaDropout struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LAlphaDropout) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLAlphaDropout{
		ClassName: "AlphaDropout",
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

func (l *LAlphaDropout) GetCustomLayerDefinition() string {
	return ``
}

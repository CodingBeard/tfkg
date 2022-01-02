package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LGaussianDropout struct {
	dtype        DataType
	inputs       []Layer
	name         string
	rate         float64
	shape        tf.Shape
	trainable    bool
	layerWeights []*tf.Tensor
}

func GaussianDropout(rate float64) *LGaussianDropout {
	return &LGaussianDropout{
		dtype:     Float32,
		name:      UniqueName("gaussian_dropout"),
		rate:      rate,
		trainable: true,
	}
}

func (l *LGaussianDropout) SetDtype(dtype DataType) *LGaussianDropout {
	l.dtype = dtype
	return l
}

func (l *LGaussianDropout) SetName(name string) *LGaussianDropout {
	l.name = name
	return l
}

func (l *LGaussianDropout) SetShape(shape tf.Shape) *LGaussianDropout {
	l.shape = shape
	return l
}

func (l *LGaussianDropout) SetTrainable(trainable bool) *LGaussianDropout {
	l.trainable = trainable
	return l
}

func (l *LGaussianDropout) SetLayerWeights(layerWeights []*tf.Tensor) *LGaussianDropout {
	l.layerWeights = layerWeights
	return l
}

func (l *LGaussianDropout) GetShape() tf.Shape {
	return l.shape
}

func (l *LGaussianDropout) GetDtype() DataType {
	return l.dtype
}

func (l *LGaussianDropout) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LGaussianDropout) GetInputs() []Layer {
	return l.inputs
}

func (l *LGaussianDropout) GetName() string {
	return l.name
}

func (l *LGaussianDropout) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLGaussianDropout struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LGaussianDropout) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLGaussianDropout{
		ClassName: "GaussianDropout",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"rate":      l.rate,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LGaussianDropout) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LGaussianNoise struct {
	dtype     DataType
	inputs    []Layer
	name      string
	shape     tf.Shape
	stddev    float64
	trainable bool
}

func GaussianNoise(stddev float64) *LGaussianNoise {
	return &LGaussianNoise{
		dtype:     Float32,
		name:      UniqueName("gaussian_noise"),
		stddev:    stddev,
		trainable: true,
	}
}

func (l *LGaussianNoise) SetDtype(dtype DataType) *LGaussianNoise {
	l.dtype = dtype
	return l
}

func (l *LGaussianNoise) SetName(name string) *LGaussianNoise {
	l.name = name
	return l
}

func (l *LGaussianNoise) SetShape(shape tf.Shape) *LGaussianNoise {
	l.shape = shape
	return l
}

func (l *LGaussianNoise) SetTrainable(trainable bool) *LGaussianNoise {
	l.trainable = trainable
	return l
}

func (l *LGaussianNoise) GetShape() tf.Shape {
	return l.shape
}

func (l *LGaussianNoise) GetDtype() DataType {
	return l.dtype
}

func (l *LGaussianNoise) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LGaussianNoise) GetInputs() []Layer {
	return l.inputs
}

func (l *LGaussianNoise) GetName() string {
	return l.name
}

type jsonConfigLGaussianNoise struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LGaussianNoise) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLGaussianNoise{
		ClassName: "GaussianNoise",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"stddev":    l.stddev,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LGaussianNoise) GetCustomLayerDefinition() string {
	return ``
}

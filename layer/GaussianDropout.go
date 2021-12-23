package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type GaussianDropout struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
	rate      float64
}

func NewGaussianDropout(rate float64, options ...GaussianDropoutOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		g := &GaussianDropout{
			rate:      rate,
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("gaussiandropout"),
		}
		for _, option := range options {
			option(g)
		}
		return g
	}
}

type GaussianDropoutOption func(*GaussianDropout)

func GaussianDropoutWithName(name string) func(g *GaussianDropout) {
	return func(g *GaussianDropout) {
		g.name = name
	}
}

func GaussianDropoutWithDtype(dtype DataType) func(g *GaussianDropout) {
	return func(g *GaussianDropout) {
		g.dtype = dtype
	}
}

func GaussianDropoutWithTrainable(trainable bool) func(g *GaussianDropout) {
	return func(g *GaussianDropout) {
		g.trainable = trainable
	}
}

func (g *GaussianDropout) GetShape() tf.Shape {
	return g.shape
}

func (g *GaussianDropout) GetDtype() DataType {
	return g.dtype
}

func (g *GaussianDropout) SetInput(inputs []Layer) {
	g.inputs = inputs
	g.dtype = inputs[0].GetDtype()
}

func (g *GaussianDropout) GetInputs() []Layer {
	return g.inputs
}

func (g *GaussianDropout) GetName() string {
	return g.name
}

type jsonConfigGaussianDropout struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (g *GaussianDropout) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range g.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigGaussianDropout{
		ClassName: "GaussianDropout",
		Name:      g.name,
		Config: map[string]interface{}{
			"dtype":     g.dtype.String(),
			"name":      g.name,
			"rate":      g.rate,
			"trainable": g.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (g *GaussianDropout) GetCustomLayerDefinition() string {
	return ``
}

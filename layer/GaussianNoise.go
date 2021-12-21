package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type GaussianNoise struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	stddev float64
}

func NewGaussianNoise(stddev float64, options ...GaussianNoiseOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		g := &GaussianNoise{
			stddev: stddev,
			trainable: true,
			inputs: inputs,
			name: uniqueName("gaussiannoise"),		
		}
		for _, option := range options {
			option(g)
		}
		return g
	}
}

type GaussianNoiseOption func (*GaussianNoise)

func GaussianNoiseWithName(name string) func(g *GaussianNoise) {
	 return func(g *GaussianNoise) {
		g.name = name
	}
}

func GaussianNoiseWithDtype(dtype DataType) func(g *GaussianNoise) {
	 return func(g *GaussianNoise) {
		g.dtype = dtype
	}
}

func GaussianNoiseWithTrainable(trainable bool) func(g *GaussianNoise) {
	 return func(g *GaussianNoise) {
		g.trainable = trainable
	}
}


func (g *GaussianNoise) GetShape() tf.Shape {
	return g.shape
}

func (g *GaussianNoise) GetDtype() DataType {
	return g.dtype
}

func (g *GaussianNoise) SetInput(inputs []Layer) {
	g.inputs = inputs
	g.dtype = inputs[0].GetDtype()
}

func (g *GaussianNoise) GetInputs() []Layer {
	return g.inputs
}

func (g *GaussianNoise) GetName() string {
	return g.name
}


type jsonConfigGaussianNoise struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (g *GaussianNoise) GetKerasLayerConfig() interface{} {
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
	return jsonConfigGaussianNoise{
		ClassName: "GaussianNoise",
		Name: g.name,
		Config: map[string]interface{}{
			"name": g.name,
			"trainable": g.trainable,
			"dtype": g.dtype.String(),
			"stddev": g.stddev,
		},
		InboundNodes: inboundNodes,
	}
}
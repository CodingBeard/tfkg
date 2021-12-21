package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type AlphaDropout struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	rate float64
	noiseShape interface{}
	seed interface{}
}

func NewAlphaDropout(rate float64, options ...AlphaDropoutOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		a := &AlphaDropout{
			rate: rate,
			noiseShape: nil,
			seed: nil,
			trainable: true,
			inputs: inputs,
			name: uniqueName("alphadropout"),		
		}
		for _, option := range options {
			option(a)
		}
		return a
	}
}

type AlphaDropoutOption func (*AlphaDropout)

func AlphaDropoutWithName(name string) func(a *AlphaDropout) {
	 return func(a *AlphaDropout) {
		a.name = name
	}
}

func AlphaDropoutWithDtype(dtype DataType) func(a *AlphaDropout) {
	 return func(a *AlphaDropout) {
		a.dtype = dtype
	}
}

func AlphaDropoutWithTrainable(trainable bool) func(a *AlphaDropout) {
	 return func(a *AlphaDropout) {
		a.trainable = trainable
	}
}

func AlphaDropoutWithNoiseShape(noiseShape interface{}) func(a *AlphaDropout) {
	 return func(a *AlphaDropout) {
		a.noiseShape = noiseShape
	}
}

func AlphaDropoutWithSeed(seed interface{}) func(a *AlphaDropout) {
	 return func(a *AlphaDropout) {
		a.seed = seed
	}
}


func (a *AlphaDropout) GetShape() tf.Shape {
	return a.shape
}

func (a *AlphaDropout) GetDtype() DataType {
	return a.dtype
}

func (a *AlphaDropout) SetInput(inputs []Layer) {
	a.inputs = inputs
	a.dtype = inputs[0].GetDtype()
}

func (a *AlphaDropout) GetInputs() []Layer {
	return a.inputs
}

func (a *AlphaDropout) GetName() string {
	return a.name
}


type jsonConfigAlphaDropout struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (a *AlphaDropout) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range a.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigAlphaDropout{
		ClassName: "AlphaDropout",
		Name: a.name,
		Config: map[string]interface{}{
			"name": a.name,
			"trainable": a.trainable,
			"dtype": a.dtype.String(),
			"rate": a.rate,
		},
		InboundNodes: inboundNodes,
	}
}
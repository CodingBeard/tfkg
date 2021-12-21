package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Average struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	
}

func NewAverage(options ...AverageOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		a := &Average{
			trainable: true,
			inputs: inputs,
			name: uniqueName("average"),		
		}
		for _, option := range options {
			option(a)
		}
		return a
	}
}

type AverageOption func (*Average)

func AverageWithName(name string) func(a *Average) {
	 return func(a *Average) {
		a.name = name
	}
}

func AverageWithDtype(dtype DataType) func(a *Average) {
	 return func(a *Average) {
		a.dtype = dtype
	}
}

func AverageWithTrainable(trainable bool) func(a *Average) {
	 return func(a *Average) {
		a.trainable = trainable
	}
}


func (a *Average) GetShape() tf.Shape {
	return a.shape
}

func (a *Average) GetDtype() DataType {
	return a.dtype
}

func (a *Average) SetInput(inputs []Layer) {
	a.inputs = inputs
	a.dtype = inputs[0].GetDtype()
}

func (a *Average) GetInputs() []Layer {
	return a.inputs
}

func (a *Average) GetName() string {
	return a.name
}


type jsonConfigAverage struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (a *Average) GetKerasLayerConfig() interface{} {
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
	return jsonConfigAverage{
		ClassName: "Average",
		Name: a.name,
		Config: map[string]interface{}{
			"name": a.name,
			"trainable": a.trainable,
			"dtype": a.dtype.String(),
		},
		InboundNodes: inboundNodes,
	}
}
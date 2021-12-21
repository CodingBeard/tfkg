package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Subtract struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	
}

func NewSubtract(options ...SubtractOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		s := &Subtract{
			trainable: true,
			inputs: inputs,
			name: uniqueName("subtract"),		
		}
		for _, option := range options {
			option(s)
		}
		return s
	}
}

type SubtractOption func (*Subtract)

func SubtractWithName(name string) func(s *Subtract) {
	 return func(s *Subtract) {
		s.name = name
	}
}

func SubtractWithDtype(dtype DataType) func(s *Subtract) {
	 return func(s *Subtract) {
		s.dtype = dtype
	}
}

func SubtractWithTrainable(trainable bool) func(s *Subtract) {
	 return func(s *Subtract) {
		s.trainable = trainable
	}
}


func (s *Subtract) GetShape() tf.Shape {
	return s.shape
}

func (s *Subtract) GetDtype() DataType {
	return s.dtype
}

func (s *Subtract) SetInput(inputs []Layer) {
	s.inputs = inputs
	s.dtype = inputs[0].GetDtype()
}

func (s *Subtract) GetInputs() []Layer {
	return s.inputs
}

func (s *Subtract) GetName() string {
	return s.name
}


type jsonConfigSubtract struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (s *Subtract) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range s.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigSubtract{
		ClassName: "Subtract",
		Name: s.name,
		Config: map[string]interface{}{
			"dtype": s.dtype.String(),
			"name": s.name,
			"trainable": s.trainable,
		},
		InboundNodes: inboundNodes,
	}
}
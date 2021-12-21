package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type RepeatVector struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	n float64
}

func NewRepeatVector(n float64, options ...RepeatVectorOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		r := &RepeatVector{
			n: n,
			trainable: true,
			inputs: inputs,
			name: uniqueName("repeatvector"),		
		}
		for _, option := range options {
			option(r)
		}
		return r
	}
}

type RepeatVectorOption func (*RepeatVector)

func RepeatVectorWithName(name string) func(r *RepeatVector) {
	 return func(r *RepeatVector) {
		r.name = name
	}
}

func RepeatVectorWithDtype(dtype DataType) func(r *RepeatVector) {
	 return func(r *RepeatVector) {
		r.dtype = dtype
	}
}

func RepeatVectorWithTrainable(trainable bool) func(r *RepeatVector) {
	 return func(r *RepeatVector) {
		r.trainable = trainable
	}
}


func (r *RepeatVector) GetShape() tf.Shape {
	return r.shape
}

func (r *RepeatVector) GetDtype() DataType {
	return r.dtype
}

func (r *RepeatVector) SetInput(inputs []Layer) {
	r.inputs = inputs
	r.dtype = inputs[0].GetDtype()
}

func (r *RepeatVector) GetInputs() []Layer {
	return r.inputs
}

func (r *RepeatVector) GetName() string {
	return r.name
}


type jsonConfigRepeatVector struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (r *RepeatVector) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range r.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigRepeatVector{
		ClassName: "RepeatVector",
		Name: r.name,
		Config: map[string]interface{}{
			"trainable": r.trainable,
			"dtype": r.dtype.String(),
			"n": r.n,
			"name": r.name,
		},
		InboundNodes: inboundNodes,
	}
}
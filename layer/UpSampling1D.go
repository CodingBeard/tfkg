package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type UpSampling1D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	size float64
}

func NewUpSampling1D(options ...UpSampling1DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		u := &UpSampling1D{
			size: 2,
			trainable: true,
			inputs: inputs,
			name: uniqueName("upsampling1d"),		
		}
		for _, option := range options {
			option(u)
		}
		return u
	}
}

type UpSampling1DOption func (*UpSampling1D)

func UpSampling1DWithName(name string) func(u *UpSampling1D) {
	 return func(u *UpSampling1D) {
		u.name = name
	}
}

func UpSampling1DWithDtype(dtype DataType) func(u *UpSampling1D) {
	 return func(u *UpSampling1D) {
		u.dtype = dtype
	}
}

func UpSampling1DWithTrainable(trainable bool) func(u *UpSampling1D) {
	 return func(u *UpSampling1D) {
		u.trainable = trainable
	}
}

func UpSampling1DWithSize(size float64) func(u *UpSampling1D) {
	 return func(u *UpSampling1D) {
		u.size = size
	}
}


func (u *UpSampling1D) GetShape() tf.Shape {
	return u.shape
}

func (u *UpSampling1D) GetDtype() DataType {
	return u.dtype
}

func (u *UpSampling1D) SetInput(inputs []Layer) {
	u.inputs = inputs
	u.dtype = inputs[0].GetDtype()
}

func (u *UpSampling1D) GetInputs() []Layer {
	return u.inputs
}

func (u *UpSampling1D) GetName() string {
	return u.name
}


type jsonConfigUpSampling1D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (u *UpSampling1D) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range u.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigUpSampling1D{
		ClassName: "UpSampling1D",
		Name: u.name,
		Config: map[string]interface{}{
			"trainable": u.trainable,
			"dtype": u.dtype.String(),
			"size": u.size,
			"name": u.name,
		},
		InboundNodes: inboundNodes,
	}
}
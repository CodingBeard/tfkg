package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type UpSampling3D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	size []interface {}
	dataFormat interface{}
}

func NewUpSampling3D(options ...UpSampling3DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		u := &UpSampling3D{
			size: []interface {}{2, 2, 2},
			dataFormat: nil,
			trainable: true,
			inputs: inputs,
			name: uniqueName("upsampling3d"),		
		}
		for _, option := range options {
			option(u)
		}
		return u
	}
}

type UpSampling3DOption func (*UpSampling3D)

func UpSampling3DWithName(name string) func(u *UpSampling3D) {
	 return func(u *UpSampling3D) {
		u.name = name
	}
}

func UpSampling3DWithDtype(dtype DataType) func(u *UpSampling3D) {
	 return func(u *UpSampling3D) {
		u.dtype = dtype
	}
}

func UpSampling3DWithTrainable(trainable bool) func(u *UpSampling3D) {
	 return func(u *UpSampling3D) {
		u.trainable = trainable
	}
}

func UpSampling3DWithSize(size []interface {}) func(u *UpSampling3D) {
	 return func(u *UpSampling3D) {
		u.size = size
	}
}

func UpSampling3DWithDataFormat(dataFormat interface{}) func(u *UpSampling3D) {
	 return func(u *UpSampling3D) {
		u.dataFormat = dataFormat
	}
}


func (u *UpSampling3D) GetShape() tf.Shape {
	return u.shape
}

func (u *UpSampling3D) GetDtype() DataType {
	return u.dtype
}

func (u *UpSampling3D) SetInput(inputs []Layer) {
	u.inputs = inputs
	u.dtype = inputs[0].GetDtype()
}

func (u *UpSampling3D) GetInputs() []Layer {
	return u.inputs
}

func (u *UpSampling3D) GetName() string {
	return u.name
}


type jsonConfigUpSampling3D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (u *UpSampling3D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigUpSampling3D{
		ClassName: "UpSampling3D",
		Name: u.name,
		Config: map[string]interface{}{
			"dtype": u.dtype.String(),
			"size": u.size,
			"data_format": u.dataFormat,
			"name": u.name,
			"trainable": u.trainable,
		},
		InboundNodes: inboundNodes,
	}
}
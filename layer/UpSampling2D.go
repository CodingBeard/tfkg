package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type UpSampling2D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	size []interface {}
	dataFormat interface{}
	interpolation string
}

func NewUpSampling2D(options ...UpSampling2DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		u := &UpSampling2D{
			size: []interface {}{2, 2},
			dataFormat: nil,
			interpolation: "nearest",
			trainable: true,
			inputs: inputs,
			name: uniqueName("upsampling2d"),		
		}
		for _, option := range options {
			option(u)
		}
		return u
	}
}

type UpSampling2DOption func (*UpSampling2D)

func UpSampling2DWithName(name string) func(u *UpSampling2D) {
	 return func(u *UpSampling2D) {
		u.name = name
	}
}

func UpSampling2DWithDtype(dtype DataType) func(u *UpSampling2D) {
	 return func(u *UpSampling2D) {
		u.dtype = dtype
	}
}

func UpSampling2DWithTrainable(trainable bool) func(u *UpSampling2D) {
	 return func(u *UpSampling2D) {
		u.trainable = trainable
	}
}

func UpSampling2DWithSize(size []interface {}) func(u *UpSampling2D) {
	 return func(u *UpSampling2D) {
		u.size = size
	}
}

func UpSampling2DWithDataFormat(dataFormat interface{}) func(u *UpSampling2D) {
	 return func(u *UpSampling2D) {
		u.dataFormat = dataFormat
	}
}

func UpSampling2DWithInterpolation(interpolation string) func(u *UpSampling2D) {
	 return func(u *UpSampling2D) {
		u.interpolation = interpolation
	}
}


func (u *UpSampling2D) GetShape() tf.Shape {
	return u.shape
}

func (u *UpSampling2D) GetDtype() DataType {
	return u.dtype
}

func (u *UpSampling2D) SetInput(inputs []Layer) {
	u.inputs = inputs
	u.dtype = inputs[0].GetDtype()
}

func (u *UpSampling2D) GetInputs() []Layer {
	return u.inputs
}

func (u *UpSampling2D) GetName() string {
	return u.name
}


type jsonConfigUpSampling2D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (u *UpSampling2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigUpSampling2D{
		ClassName: "UpSampling2D",
		Name: u.name,
		Config: map[string]interface{}{
			"name": u.name,
			"trainable": u.trainable,
			"dtype": u.dtype.String(),
			"size": u.size,
			"data_format": u.dataFormat,
			"interpolation": u.interpolation,
		},
		InboundNodes: inboundNodes,
	}
}
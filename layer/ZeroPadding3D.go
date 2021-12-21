package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type ZeroPadding3D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	padding []interface {}
	dataFormat interface{}
}

func NewZeroPadding3D(options ...ZeroPadding3DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		z := &ZeroPadding3D{
			padding: []interface {}{1, 1, 1},
			dataFormat: nil,
			trainable: true,
			inputs: inputs,
			name: uniqueName("zeropadding3d"),		
		}
		for _, option := range options {
			option(z)
		}
		return z
	}
}

type ZeroPadding3DOption func (*ZeroPadding3D)

func ZeroPadding3DWithName(name string) func(z *ZeroPadding3D) {
	 return func(z *ZeroPadding3D) {
		z.name = name
	}
}

func ZeroPadding3DWithDtype(dtype DataType) func(z *ZeroPadding3D) {
	 return func(z *ZeroPadding3D) {
		z.dtype = dtype
	}
}

func ZeroPadding3DWithTrainable(trainable bool) func(z *ZeroPadding3D) {
	 return func(z *ZeroPadding3D) {
		z.trainable = trainable
	}
}

func ZeroPadding3DWithPadding(padding []interface {}) func(z *ZeroPadding3D) {
	 return func(z *ZeroPadding3D) {
		z.padding = padding
	}
}

func ZeroPadding3DWithDataFormat(dataFormat interface{}) func(z *ZeroPadding3D) {
	 return func(z *ZeroPadding3D) {
		z.dataFormat = dataFormat
	}
}


func (z *ZeroPadding3D) GetShape() tf.Shape {
	return z.shape
}

func (z *ZeroPadding3D) GetDtype() DataType {
	return z.dtype
}

func (z *ZeroPadding3D) SetInput(inputs []Layer) {
	z.inputs = inputs
	z.dtype = inputs[0].GetDtype()
}

func (z *ZeroPadding3D) GetInputs() []Layer {
	return z.inputs
}

func (z *ZeroPadding3D) GetName() string {
	return z.name
}


type jsonConfigZeroPadding3D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (z *ZeroPadding3D) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range z.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigZeroPadding3D{
		ClassName: "ZeroPadding3D",
		Name: z.name,
		Config: map[string]interface{}{
			"name": z.name,
			"trainable": z.trainable,
			"dtype": z.dtype.String(),
			"padding": z.padding,
			"data_format": z.dataFormat,
		},
		InboundNodes: inboundNodes,
	}
}
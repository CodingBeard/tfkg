package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type ZeroPadding2D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	padding []interface {}
	dataFormat interface{}
}

func NewZeroPadding2D(options ...ZeroPadding2DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		z := &ZeroPadding2D{
			padding: []interface {}{1, 1},
			dataFormat: nil,
			trainable: true,
			inputs: inputs,
			name: uniqueName("zeropadding2d"),		
		}
		for _, option := range options {
			option(z)
		}
		return z
	}
}

type ZeroPadding2DOption func (*ZeroPadding2D)

func ZeroPadding2DWithName(name string) func(z *ZeroPadding2D) {
	 return func(z *ZeroPadding2D) {
		z.name = name
	}
}

func ZeroPadding2DWithDtype(dtype DataType) func(z *ZeroPadding2D) {
	 return func(z *ZeroPadding2D) {
		z.dtype = dtype
	}
}

func ZeroPadding2DWithTrainable(trainable bool) func(z *ZeroPadding2D) {
	 return func(z *ZeroPadding2D) {
		z.trainable = trainable
	}
}

func ZeroPadding2DWithPadding(padding []interface {}) func(z *ZeroPadding2D) {
	 return func(z *ZeroPadding2D) {
		z.padding = padding
	}
}

func ZeroPadding2DWithDataFormat(dataFormat interface{}) func(z *ZeroPadding2D) {
	 return func(z *ZeroPadding2D) {
		z.dataFormat = dataFormat
	}
}


func (z *ZeroPadding2D) GetShape() tf.Shape {
	return z.shape
}

func (z *ZeroPadding2D) GetDtype() DataType {
	return z.dtype
}

func (z *ZeroPadding2D) SetInput(inputs []Layer) {
	z.inputs = inputs
	z.dtype = inputs[0].GetDtype()
}

func (z *ZeroPadding2D) GetInputs() []Layer {
	return z.inputs
}

func (z *ZeroPadding2D) GetName() string {
	return z.name
}


type jsonConfigZeroPadding2D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (z *ZeroPadding2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigZeroPadding2D{
		ClassName: "ZeroPadding2D",
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
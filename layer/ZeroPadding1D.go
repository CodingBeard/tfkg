package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type ZeroPadding1D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	padding float64
}

func NewZeroPadding1D(options ...ZeroPadding1DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		z := &ZeroPadding1D{
			padding: 1,
			trainable: true,
			inputs: inputs,
			name: uniqueName("zeropadding1d"),		
		}
		for _, option := range options {
			option(z)
		}
		return z
	}
}

type ZeroPadding1DOption func (*ZeroPadding1D)

func ZeroPadding1DWithName(name string) func(z *ZeroPadding1D) {
	 return func(z *ZeroPadding1D) {
		z.name = name
	}
}

func ZeroPadding1DWithDtype(dtype DataType) func(z *ZeroPadding1D) {
	 return func(z *ZeroPadding1D) {
		z.dtype = dtype
	}
}

func ZeroPadding1DWithTrainable(trainable bool) func(z *ZeroPadding1D) {
	 return func(z *ZeroPadding1D) {
		z.trainable = trainable
	}
}

func ZeroPadding1DWithPadding(padding float64) func(z *ZeroPadding1D) {
	 return func(z *ZeroPadding1D) {
		z.padding = padding
	}
}


func (z *ZeroPadding1D) GetShape() tf.Shape {
	return z.shape
}

func (z *ZeroPadding1D) GetDtype() DataType {
	return z.dtype
}

func (z *ZeroPadding1D) SetInput(inputs []Layer) {
	z.inputs = inputs
	z.dtype = inputs[0].GetDtype()
}

func (z *ZeroPadding1D) GetInputs() []Layer {
	return z.inputs
}

func (z *ZeroPadding1D) GetName() string {
	return z.name
}


type jsonConfigZeroPadding1D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (z *ZeroPadding1D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigZeroPadding1D{
		ClassName: "ZeroPadding1D",
		Name: z.name,
		Config: map[string]interface{}{
			"name": z.name,
			"trainable": z.trainable,
			"dtype": z.dtype.String(),
			"padding": z.padding,
		},
		InboundNodes: inboundNodes,
	}
}
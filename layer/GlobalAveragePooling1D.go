package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type GlobalAveragePooling1D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	dataFormat string
	keepdims bool
}

func NewGlobalAveragePooling1D(options ...GlobalAveragePooling1DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		g := &GlobalAveragePooling1D{
			dataFormat: "channels_last",
			keepdims: false,
			trainable: true,
			inputs: inputs,
			name: uniqueName("globalaveragepooling1d"),		
		}
		for _, option := range options {
			option(g)
		}
		return g
	}
}

type GlobalAveragePooling1DOption func (*GlobalAveragePooling1D)

func GlobalAveragePooling1DWithName(name string) func(g *GlobalAveragePooling1D) {
	 return func(g *GlobalAveragePooling1D) {
		g.name = name
	}
}

func GlobalAveragePooling1DWithDtype(dtype DataType) func(g *GlobalAveragePooling1D) {
	 return func(g *GlobalAveragePooling1D) {
		g.dtype = dtype
	}
}

func GlobalAveragePooling1DWithTrainable(trainable bool) func(g *GlobalAveragePooling1D) {
	 return func(g *GlobalAveragePooling1D) {
		g.trainable = trainable
	}
}

func GlobalAveragePooling1DWithDataFormat(dataFormat string) func(g *GlobalAveragePooling1D) {
	 return func(g *GlobalAveragePooling1D) {
		g.dataFormat = dataFormat
	}
}


func (g *GlobalAveragePooling1D) GetShape() tf.Shape {
	return g.shape
}

func (g *GlobalAveragePooling1D) GetDtype() DataType {
	return g.dtype
}

func (g *GlobalAveragePooling1D) SetInput(inputs []Layer) {
	g.inputs = inputs
	g.dtype = inputs[0].GetDtype()
}

func (g *GlobalAveragePooling1D) GetInputs() []Layer {
	return g.inputs
}

func (g *GlobalAveragePooling1D) GetName() string {
	return g.name
}


type jsonConfigGlobalAveragePooling1D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (g *GlobalAveragePooling1D) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range g.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigGlobalAveragePooling1D{
		ClassName: "GlobalAveragePooling1D",
		Name: g.name,
		Config: map[string]interface{}{
			"data_format": g.dataFormat,
			"keepdims": g.keepdims,
			"name": g.name,
			"trainable": g.trainable,
			"dtype": g.dtype.String(),
		},
		InboundNodes: inboundNodes,
	}
}
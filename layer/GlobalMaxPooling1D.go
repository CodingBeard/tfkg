package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type GlobalMaxPooling1D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	dataFormat string
	keepdims bool
}

func NewGlobalMaxPooling1D(options ...GlobalMaxPooling1DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		g := &GlobalMaxPooling1D{
			dataFormat: "channels_last",
			keepdims: false,
			trainable: true,
			inputs: inputs,
			name: uniqueName("globalmaxpooling1d"),		
		}
		for _, option := range options {
			option(g)
		}
		return g
	}
}

type GlobalMaxPooling1DOption func (*GlobalMaxPooling1D)

func GlobalMaxPooling1DWithName(name string) func(g *GlobalMaxPooling1D) {
	 return func(g *GlobalMaxPooling1D) {
		g.name = name
	}
}

func GlobalMaxPooling1DWithDtype(dtype DataType) func(g *GlobalMaxPooling1D) {
	 return func(g *GlobalMaxPooling1D) {
		g.dtype = dtype
	}
}

func GlobalMaxPooling1DWithTrainable(trainable bool) func(g *GlobalMaxPooling1D) {
	 return func(g *GlobalMaxPooling1D) {
		g.trainable = trainable
	}
}

func GlobalMaxPooling1DWithDataFormat(dataFormat string) func(g *GlobalMaxPooling1D) {
	 return func(g *GlobalMaxPooling1D) {
		g.dataFormat = dataFormat
	}
}

func GlobalMaxPooling1DWithKeepdims(keepdims bool) func(g *GlobalMaxPooling1D) {
	 return func(g *GlobalMaxPooling1D) {
		g.keepdims = keepdims
	}
}


func (g *GlobalMaxPooling1D) GetShape() tf.Shape {
	return g.shape
}

func (g *GlobalMaxPooling1D) GetDtype() DataType {
	return g.dtype
}

func (g *GlobalMaxPooling1D) SetInput(inputs []Layer) {
	g.inputs = inputs
	g.dtype = inputs[0].GetDtype()
}

func (g *GlobalMaxPooling1D) GetInputs() []Layer {
	return g.inputs
}

func (g *GlobalMaxPooling1D) GetName() string {
	return g.name
}


type jsonConfigGlobalMaxPooling1D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (g *GlobalMaxPooling1D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigGlobalMaxPooling1D{
		ClassName: "GlobalMaxPooling1D",
		Name: g.name,
		Config: map[string]interface{}{
			"dtype": g.dtype.String(),
			"data_format": g.dataFormat,
			"keepdims": g.keepdims,
			"name": g.name,
			"trainable": g.trainable,
		},
		InboundNodes: inboundNodes,
	}
}
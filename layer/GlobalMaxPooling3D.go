package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type GlobalMaxPooling3D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	dataFormat interface{}
	keepdims bool
}

func NewGlobalMaxPooling3D(options ...GlobalMaxPooling3DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		g := &GlobalMaxPooling3D{
			dataFormat: nil,
			keepdims: false,
			trainable: true,
			inputs: inputs,
			name: uniqueName("globalmaxpooling3d"),		
		}
		for _, option := range options {
			option(g)
		}
		return g
	}
}

type GlobalMaxPooling3DOption func (*GlobalMaxPooling3D)

func GlobalMaxPooling3DWithName(name string) func(g *GlobalMaxPooling3D) {
	 return func(g *GlobalMaxPooling3D) {
		g.name = name
	}
}

func GlobalMaxPooling3DWithDtype(dtype DataType) func(g *GlobalMaxPooling3D) {
	 return func(g *GlobalMaxPooling3D) {
		g.dtype = dtype
	}
}

func GlobalMaxPooling3DWithTrainable(trainable bool) func(g *GlobalMaxPooling3D) {
	 return func(g *GlobalMaxPooling3D) {
		g.trainable = trainable
	}
}

func GlobalMaxPooling3DWithDataFormat(dataFormat interface{}) func(g *GlobalMaxPooling3D) {
	 return func(g *GlobalMaxPooling3D) {
		g.dataFormat = dataFormat
	}
}

func GlobalMaxPooling3DWithKeepdims(keepdims bool) func(g *GlobalMaxPooling3D) {
	 return func(g *GlobalMaxPooling3D) {
		g.keepdims = keepdims
	}
}


func (g *GlobalMaxPooling3D) GetShape() tf.Shape {
	return g.shape
}

func (g *GlobalMaxPooling3D) GetDtype() DataType {
	return g.dtype
}

func (g *GlobalMaxPooling3D) SetInput(inputs []Layer) {
	g.inputs = inputs
	g.dtype = inputs[0].GetDtype()
}

func (g *GlobalMaxPooling3D) GetInputs() []Layer {
	return g.inputs
}

func (g *GlobalMaxPooling3D) GetName() string {
	return g.name
}


type jsonConfigGlobalMaxPooling3D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (g *GlobalMaxPooling3D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigGlobalMaxPooling3D{
		ClassName: "GlobalMaxPooling3D",
		Name: g.name,
		Config: map[string]interface{}{
			"keepdims": g.keepdims,
			"name": g.name,
			"trainable": g.trainable,
			"dtype": g.dtype.String(),
			"data_format": g.dataFormat,
		},
		InboundNodes: inboundNodes,
	}
}
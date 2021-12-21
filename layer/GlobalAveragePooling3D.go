package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type GlobalAveragePooling3D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	dataFormat interface{}
	keepdims bool
}

func NewGlobalAveragePooling3D(options ...GlobalAveragePooling3DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		g := &GlobalAveragePooling3D{
			dataFormat: nil,
			keepdims: false,
			trainable: true,
			inputs: inputs,
			name: uniqueName("globalaveragepooling3d"),		
		}
		for _, option := range options {
			option(g)
		}
		return g
	}
}

type GlobalAveragePooling3DOption func (*GlobalAveragePooling3D)

func GlobalAveragePooling3DWithName(name string) func(g *GlobalAveragePooling3D) {
	 return func(g *GlobalAveragePooling3D) {
		g.name = name
	}
}

func GlobalAveragePooling3DWithDtype(dtype DataType) func(g *GlobalAveragePooling3D) {
	 return func(g *GlobalAveragePooling3D) {
		g.dtype = dtype
	}
}

func GlobalAveragePooling3DWithTrainable(trainable bool) func(g *GlobalAveragePooling3D) {
	 return func(g *GlobalAveragePooling3D) {
		g.trainable = trainable
	}
}

func GlobalAveragePooling3DWithDataFormat(dataFormat interface{}) func(g *GlobalAveragePooling3D) {
	 return func(g *GlobalAveragePooling3D) {
		g.dataFormat = dataFormat
	}
}

func GlobalAveragePooling3DWithKeepdims(keepdims bool) func(g *GlobalAveragePooling3D) {
	 return func(g *GlobalAveragePooling3D) {
		g.keepdims = keepdims
	}
}


func (g *GlobalAveragePooling3D) GetShape() tf.Shape {
	return g.shape
}

func (g *GlobalAveragePooling3D) GetDtype() DataType {
	return g.dtype
}

func (g *GlobalAveragePooling3D) SetInput(inputs []Layer) {
	g.inputs = inputs
	g.dtype = inputs[0].GetDtype()
}

func (g *GlobalAveragePooling3D) GetInputs() []Layer {
	return g.inputs
}

func (g *GlobalAveragePooling3D) GetName() string {
	return g.name
}


type jsonConfigGlobalAveragePooling3D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (g *GlobalAveragePooling3D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigGlobalAveragePooling3D{
		ClassName: "GlobalAveragePooling3D",
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
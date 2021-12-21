package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type GlobalAveragePooling2D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	dataFormat interface{}
	keepdims bool
}

func NewGlobalAveragePooling2D(options ...GlobalAveragePooling2DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		g := &GlobalAveragePooling2D{
			dataFormat: nil,
			keepdims: false,
			trainable: true,
			inputs: inputs,
			name: uniqueName("globalaveragepooling2d"),		
		}
		for _, option := range options {
			option(g)
		}
		return g
	}
}

type GlobalAveragePooling2DOption func (*GlobalAveragePooling2D)

func GlobalAveragePooling2DWithName(name string) func(g *GlobalAveragePooling2D) {
	 return func(g *GlobalAveragePooling2D) {
		g.name = name
	}
}

func GlobalAveragePooling2DWithDtype(dtype DataType) func(g *GlobalAveragePooling2D) {
	 return func(g *GlobalAveragePooling2D) {
		g.dtype = dtype
	}
}

func GlobalAveragePooling2DWithTrainable(trainable bool) func(g *GlobalAveragePooling2D) {
	 return func(g *GlobalAveragePooling2D) {
		g.trainable = trainable
	}
}

func GlobalAveragePooling2DWithDataFormat(dataFormat interface{}) func(g *GlobalAveragePooling2D) {
	 return func(g *GlobalAveragePooling2D) {
		g.dataFormat = dataFormat
	}
}

func GlobalAveragePooling2DWithKeepdims(keepdims bool) func(g *GlobalAveragePooling2D) {
	 return func(g *GlobalAveragePooling2D) {
		g.keepdims = keepdims
	}
}


func (g *GlobalAveragePooling2D) GetShape() tf.Shape {
	return g.shape
}

func (g *GlobalAveragePooling2D) GetDtype() DataType {
	return g.dtype
}

func (g *GlobalAveragePooling2D) SetInput(inputs []Layer) {
	g.inputs = inputs
	g.dtype = inputs[0].GetDtype()
}

func (g *GlobalAveragePooling2D) GetInputs() []Layer {
	return g.inputs
}

func (g *GlobalAveragePooling2D) GetName() string {
	return g.name
}


type jsonConfigGlobalAveragePooling2D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (g *GlobalAveragePooling2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigGlobalAveragePooling2D{
		ClassName: "GlobalAveragePooling2D",
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
package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type GlobalMaxPooling2D struct {
	name       string
	dtype      DataType
	inputs     []Layer
	shape      tf.Shape
	trainable  bool
	dataFormat interface{}
	keepdims   bool
}

func NewGlobalMaxPooling2D(options ...GlobalMaxPooling2DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		g := &GlobalMaxPooling2D{
			dataFormat: nil,
			keepdims:   false,
			trainable:  true,
			inputs:     inputs,
			name:       UniqueName("globalmaxpooling2d"),
		}
		for _, option := range options {
			option(g)
		}
		return g
	}
}

type GlobalMaxPooling2DOption func(*GlobalMaxPooling2D)

func GlobalMaxPooling2DWithName(name string) func(g *GlobalMaxPooling2D) {
	return func(g *GlobalMaxPooling2D) {
		g.name = name
	}
}

func GlobalMaxPooling2DWithDtype(dtype DataType) func(g *GlobalMaxPooling2D) {
	return func(g *GlobalMaxPooling2D) {
		g.dtype = dtype
	}
}

func GlobalMaxPooling2DWithTrainable(trainable bool) func(g *GlobalMaxPooling2D) {
	return func(g *GlobalMaxPooling2D) {
		g.trainable = trainable
	}
}

func GlobalMaxPooling2DWithDataFormat(dataFormat interface{}) func(g *GlobalMaxPooling2D) {
	return func(g *GlobalMaxPooling2D) {
		g.dataFormat = dataFormat
	}
}

func GlobalMaxPooling2DWithKeepdims(keepdims bool) func(g *GlobalMaxPooling2D) {
	return func(g *GlobalMaxPooling2D) {
		g.keepdims = keepdims
	}
}

func (g *GlobalMaxPooling2D) GetShape() tf.Shape {
	return g.shape
}

func (g *GlobalMaxPooling2D) GetDtype() DataType {
	return g.dtype
}

func (g *GlobalMaxPooling2D) SetInput(inputs []Layer) {
	g.inputs = inputs
	g.dtype = inputs[0].GetDtype()
}

func (g *GlobalMaxPooling2D) GetInputs() []Layer {
	return g.inputs
}

func (g *GlobalMaxPooling2D) GetName() string {
	return g.name
}

type jsonConfigGlobalMaxPooling2D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (g *GlobalMaxPooling2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigGlobalMaxPooling2D{
		ClassName: "GlobalMaxPooling2D",
		Name:      g.name,
		Config: map[string]interface{}{
			"data_format": g.dataFormat,
			"dtype":       g.dtype.String(),
			"keepdims":    g.keepdims,
			"name":        g.name,
			"trainable":   g.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (g *GlobalMaxPooling2D) GetCustomLayerDefinition() string {
	return ``
}

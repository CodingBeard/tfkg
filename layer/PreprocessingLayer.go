package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type PreprocessingLayer struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
}

func NewPreprocessingLayer(options ...PreprocessingLayerOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		p := &PreprocessingLayer{
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("preprocessinglayer"),
		}
		for _, option := range options {
			option(p)
		}
		return p
	}
}

type PreprocessingLayerOption func(*PreprocessingLayer)

func PreprocessingLayerWithName(name string) func(p *PreprocessingLayer) {
	return func(p *PreprocessingLayer) {
		p.name = name
	}
}

func PreprocessingLayerWithDtype(dtype DataType) func(p *PreprocessingLayer) {
	return func(p *PreprocessingLayer) {
		p.dtype = dtype
	}
}

func PreprocessingLayerWithTrainable(trainable bool) func(p *PreprocessingLayer) {
	return func(p *PreprocessingLayer) {
		p.trainable = trainable
	}
}

func (p *PreprocessingLayer) GetShape() tf.Shape {
	return p.shape
}

func (p *PreprocessingLayer) GetDtype() DataType {
	return p.dtype
}

func (p *PreprocessingLayer) SetInput(inputs []Layer) {
	p.inputs = inputs
	p.dtype = inputs[0].GetDtype()
}

func (p *PreprocessingLayer) GetInputs() []Layer {
	return p.inputs
}

func (p *PreprocessingLayer) GetName() string {
	return p.name
}

type jsonConfigPreprocessingLayer struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (p *PreprocessingLayer) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range p.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigPreprocessingLayer{
		ClassName: "PreprocessingLayer",
		Name:      p.name,
		Config: map[string]interface{}{
			"dtype":     p.dtype.String(),
			"name":      p.name,
			"trainable": p.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (p *PreprocessingLayer) GetCustomLayerDefinition() string {
	return ``
}

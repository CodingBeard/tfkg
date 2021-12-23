package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Flatten struct {
	name       string
	dtype      DataType
	inputs     []Layer
	shape      tf.Shape
	trainable  bool
	dataFormat interface{}
}

func NewFlatten(options ...FlattenOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		f := &Flatten{
			dataFormat: nil,
			trainable:  true,
			inputs:     inputs,
			name:       UniqueName("flatten"),
		}
		for _, option := range options {
			option(f)
		}
		return f
	}
}

type FlattenOption func(*Flatten)

func FlattenWithName(name string) func(f *Flatten) {
	return func(f *Flatten) {
		f.name = name
	}
}

func FlattenWithDtype(dtype DataType) func(f *Flatten) {
	return func(f *Flatten) {
		f.dtype = dtype
	}
}

func FlattenWithTrainable(trainable bool) func(f *Flatten) {
	return func(f *Flatten) {
		f.trainable = trainable
	}
}

func FlattenWithDataFormat(dataFormat interface{}) func(f *Flatten) {
	return func(f *Flatten) {
		f.dataFormat = dataFormat
	}
}

func (f *Flatten) GetShape() tf.Shape {
	return f.shape
}

func (f *Flatten) GetDtype() DataType {
	return f.dtype
}

func (f *Flatten) SetInput(inputs []Layer) {
	f.inputs = inputs
	f.dtype = inputs[0].GetDtype()
}

func (f *Flatten) GetInputs() []Layer {
	return f.inputs
}

func (f *Flatten) GetName() string {
	return f.name
}

type jsonConfigFlatten struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (f *Flatten) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range f.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigFlatten{
		ClassName: "Flatten",
		Name:      f.name,
		Config: map[string]interface{}{
			"data_format": f.dataFormat,
			"dtype":       f.dtype.String(),
			"name":        f.name,
			"trainable":   f.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (f *Flatten) GetCustomLayerDefinition() string {
	return ``
}

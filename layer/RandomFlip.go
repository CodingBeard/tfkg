package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type RandomFlip struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
	mode      string
	seed      interface{}
}

func NewRandomFlip(options ...RandomFlipOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		r := &RandomFlip{
			mode:      "horizontal_and_vertical",
			seed:      nil,
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("randomflip"),
		}
		for _, option := range options {
			option(r)
		}
		return r
	}
}

type RandomFlipOption func(*RandomFlip)

func RandomFlipWithName(name string) func(r *RandomFlip) {
	return func(r *RandomFlip) {
		r.name = name
	}
}

func RandomFlipWithDtype(dtype DataType) func(r *RandomFlip) {
	return func(r *RandomFlip) {
		r.dtype = dtype
	}
}

func RandomFlipWithTrainable(trainable bool) func(r *RandomFlip) {
	return func(r *RandomFlip) {
		r.trainable = trainable
	}
}

func RandomFlipWithMode(mode string) func(r *RandomFlip) {
	return func(r *RandomFlip) {
		r.mode = mode
	}
}

func RandomFlipWithSeed(seed interface{}) func(r *RandomFlip) {
	return func(r *RandomFlip) {
		r.seed = seed
	}
}

func (r *RandomFlip) GetShape() tf.Shape {
	return r.shape
}

func (r *RandomFlip) GetDtype() DataType {
	return r.dtype
}

func (r *RandomFlip) SetInput(inputs []Layer) {
	r.inputs = inputs
	r.dtype = inputs[0].GetDtype()
}

func (r *RandomFlip) GetInputs() []Layer {
	return r.inputs
}

func (r *RandomFlip) GetName() string {
	return r.name
}

type jsonConfigRandomFlip struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (r *RandomFlip) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range r.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigRandomFlip{
		ClassName: "RandomFlip",
		Name:      r.name,
		Config: map[string]interface{}{
			"dtype":     r.dtype.String(),
			"mode":      r.mode,
			"name":      r.name,
			"seed":      r.seed,
			"trainable": r.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (r *RandomFlip) GetCustomLayerDefinition() string {
	return ``
}

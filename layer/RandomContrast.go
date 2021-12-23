package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type RandomContrast struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
	factor    float64
	seed      interface{}
}

func NewRandomContrast(factor float64, options ...RandomContrastOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		r := &RandomContrast{
			factor:    factor,
			seed:      nil,
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("randomcontrast"),
		}
		for _, option := range options {
			option(r)
		}
		return r
	}
}

type RandomContrastOption func(*RandomContrast)

func RandomContrastWithName(name string) func(r *RandomContrast) {
	return func(r *RandomContrast) {
		r.name = name
	}
}

func RandomContrastWithDtype(dtype DataType) func(r *RandomContrast) {
	return func(r *RandomContrast) {
		r.dtype = dtype
	}
}

func RandomContrastWithTrainable(trainable bool) func(r *RandomContrast) {
	return func(r *RandomContrast) {
		r.trainable = trainable
	}
}

func RandomContrastWithSeed(seed interface{}) func(r *RandomContrast) {
	return func(r *RandomContrast) {
		r.seed = seed
	}
}

func (r *RandomContrast) GetShape() tf.Shape {
	return r.shape
}

func (r *RandomContrast) GetDtype() DataType {
	return r.dtype
}

func (r *RandomContrast) SetInput(inputs []Layer) {
	r.inputs = inputs
	r.dtype = inputs[0].GetDtype()
}

func (r *RandomContrast) GetInputs() []Layer {
	return r.inputs
}

func (r *RandomContrast) GetName() string {
	return r.name
}

type jsonConfigRandomContrast struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (r *RandomContrast) GetKerasLayerConfig() interface{} {
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
	return jsonConfigRandomContrast{
		ClassName: "RandomContrast",
		Name:      r.name,
		Config: map[string]interface{}{
			"dtype":     r.dtype.String(),
			"factor":    r.factor,
			"name":      r.name,
			"seed":      r.seed,
			"trainable": r.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (r *RandomContrast) GetCustomLayerDefinition() string {
	return ``
}

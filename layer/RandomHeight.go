package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type RandomHeight struct {
	name          string
	dtype         DataType
	inputs        []Layer
	shape         tf.Shape
	trainable     bool
	factor        float64
	interpolation string
	seed          interface{}
}

func NewRandomHeight(factor float64, options ...RandomHeightOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		r := &RandomHeight{
			factor:        factor,
			interpolation: "bilinear",
			seed:          nil,
			trainable:     true,
			inputs:        inputs,
			name:          UniqueName("randomheight"),
		}
		for _, option := range options {
			option(r)
		}
		return r
	}
}

type RandomHeightOption func(*RandomHeight)

func RandomHeightWithName(name string) func(r *RandomHeight) {
	return func(r *RandomHeight) {
		r.name = name
	}
}

func RandomHeightWithDtype(dtype DataType) func(r *RandomHeight) {
	return func(r *RandomHeight) {
		r.dtype = dtype
	}
}

func RandomHeightWithTrainable(trainable bool) func(r *RandomHeight) {
	return func(r *RandomHeight) {
		r.trainable = trainable
	}
}

func RandomHeightWithInterpolation(interpolation string) func(r *RandomHeight) {
	return func(r *RandomHeight) {
		r.interpolation = interpolation
	}
}

func RandomHeightWithSeed(seed interface{}) func(r *RandomHeight) {
	return func(r *RandomHeight) {
		r.seed = seed
	}
}

func (r *RandomHeight) GetShape() tf.Shape {
	return r.shape
}

func (r *RandomHeight) GetDtype() DataType {
	return r.dtype
}

func (r *RandomHeight) SetInput(inputs []Layer) {
	r.inputs = inputs
	r.dtype = inputs[0].GetDtype()
}

func (r *RandomHeight) GetInputs() []Layer {
	return r.inputs
}

func (r *RandomHeight) GetName() string {
	return r.name
}

type jsonConfigRandomHeight struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (r *RandomHeight) GetKerasLayerConfig() interface{} {
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
	return jsonConfigRandomHeight{
		ClassName: "RandomHeight",
		Name:      r.name,
		Config: map[string]interface{}{
			"dtype":         r.dtype.String(),
			"factor":        r.factor,
			"interpolation": r.interpolation,
			"name":          r.name,
			"seed":          r.seed,
			"trainable":     r.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (r *RandomHeight) GetCustomLayerDefinition() string {
	return ``
}

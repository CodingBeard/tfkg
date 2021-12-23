package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type RandomWidth struct {
	name          string
	dtype         DataType
	inputs        []Layer
	shape         tf.Shape
	trainable     bool
	factor        float64
	interpolation string
	seed          interface{}
}

func NewRandomWidth(factor float64, options ...RandomWidthOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		r := &RandomWidth{
			factor:        factor,
			interpolation: "bilinear",
			seed:          nil,
			trainable:     true,
			inputs:        inputs,
			name:          UniqueName("randomwidth"),
		}
		for _, option := range options {
			option(r)
		}
		return r
	}
}

type RandomWidthOption func(*RandomWidth)

func RandomWidthWithName(name string) func(r *RandomWidth) {
	return func(r *RandomWidth) {
		r.name = name
	}
}

func RandomWidthWithDtype(dtype DataType) func(r *RandomWidth) {
	return func(r *RandomWidth) {
		r.dtype = dtype
	}
}

func RandomWidthWithTrainable(trainable bool) func(r *RandomWidth) {
	return func(r *RandomWidth) {
		r.trainable = trainable
	}
}

func RandomWidthWithInterpolation(interpolation string) func(r *RandomWidth) {
	return func(r *RandomWidth) {
		r.interpolation = interpolation
	}
}

func RandomWidthWithSeed(seed interface{}) func(r *RandomWidth) {
	return func(r *RandomWidth) {
		r.seed = seed
	}
}

func (r *RandomWidth) GetShape() tf.Shape {
	return r.shape
}

func (r *RandomWidth) GetDtype() DataType {
	return r.dtype
}

func (r *RandomWidth) SetInput(inputs []Layer) {
	r.inputs = inputs
	r.dtype = inputs[0].GetDtype()
}

func (r *RandomWidth) GetInputs() []Layer {
	return r.inputs
}

func (r *RandomWidth) GetName() string {
	return r.name
}

type jsonConfigRandomWidth struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (r *RandomWidth) GetKerasLayerConfig() interface{} {
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
	return jsonConfigRandomWidth{
		ClassName: "RandomWidth",
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

func (r *RandomWidth) GetCustomLayerDefinition() string {
	return ``
}

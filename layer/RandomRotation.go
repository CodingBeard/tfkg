package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type RandomRotation struct {
	name          string
	dtype         DataType
	inputs        []Layer
	shape         tf.Shape
	trainable     bool
	factor        float64
	fillMode      string
	interpolation string
	seed          interface{}
	fillValue     float64
}

func NewRandomRotation(factor float64, options ...RandomRotationOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		r := &RandomRotation{
			factor:        factor,
			fillMode:      "reflect",
			interpolation: "bilinear",
			seed:          nil,
			fillValue:     0,
			trainable:     true,
			inputs:        inputs,
			name:          UniqueName("randomrotation"),
		}
		for _, option := range options {
			option(r)
		}
		return r
	}
}

type RandomRotationOption func(*RandomRotation)

func RandomRotationWithName(name string) func(r *RandomRotation) {
	return func(r *RandomRotation) {
		r.name = name
	}
}

func RandomRotationWithDtype(dtype DataType) func(r *RandomRotation) {
	return func(r *RandomRotation) {
		r.dtype = dtype
	}
}

func RandomRotationWithTrainable(trainable bool) func(r *RandomRotation) {
	return func(r *RandomRotation) {
		r.trainable = trainable
	}
}

func RandomRotationWithFillMode(fillMode string) func(r *RandomRotation) {
	return func(r *RandomRotation) {
		r.fillMode = fillMode
	}
}

func RandomRotationWithInterpolation(interpolation string) func(r *RandomRotation) {
	return func(r *RandomRotation) {
		r.interpolation = interpolation
	}
}

func RandomRotationWithSeed(seed interface{}) func(r *RandomRotation) {
	return func(r *RandomRotation) {
		r.seed = seed
	}
}

func RandomRotationWithFillValue(fillValue float64) func(r *RandomRotation) {
	return func(r *RandomRotation) {
		r.fillValue = fillValue
	}
}

func (r *RandomRotation) GetShape() tf.Shape {
	return r.shape
}

func (r *RandomRotation) GetDtype() DataType {
	return r.dtype
}

func (r *RandomRotation) SetInput(inputs []Layer) {
	r.inputs = inputs
	r.dtype = inputs[0].GetDtype()
}

func (r *RandomRotation) GetInputs() []Layer {
	return r.inputs
}

func (r *RandomRotation) GetName() string {
	return r.name
}

type jsonConfigRandomRotation struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (r *RandomRotation) GetKerasLayerConfig() interface{} {
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
	return jsonConfigRandomRotation{
		ClassName: "RandomRotation",
		Name:      r.name,
		Config: map[string]interface{}{
			"dtype":         r.dtype.String(),
			"factor":        r.factor,
			"fill_mode":     r.fillMode,
			"fill_value":    r.fillValue,
			"interpolation": r.interpolation,
			"name":          r.name,
			"seed":          r.seed,
			"trainable":     r.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (r *RandomRotation) GetCustomLayerDefinition() string {
	return ``
}

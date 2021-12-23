package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type RandomTranslation struct {
	name          string
	dtype         DataType
	inputs        []Layer
	shape         tf.Shape
	trainable     bool
	heightFactor  float64
	widthFactor   float64
	fillMode      string
	interpolation string
	seed          interface{}
	fillValue     float64
}

func NewRandomTranslation(heightFactor float64, widthFactor float64, options ...RandomTranslationOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		r := &RandomTranslation{
			heightFactor:  heightFactor,
			widthFactor:   widthFactor,
			fillMode:      "reflect",
			interpolation: "bilinear",
			seed:          nil,
			fillValue:     0,
			trainable:     true,
			inputs:        inputs,
			name:          UniqueName("randomtranslation"),
		}
		for _, option := range options {
			option(r)
		}
		return r
	}
}

type RandomTranslationOption func(*RandomTranslation)

func RandomTranslationWithName(name string) func(r *RandomTranslation) {
	return func(r *RandomTranslation) {
		r.name = name
	}
}

func RandomTranslationWithDtype(dtype DataType) func(r *RandomTranslation) {
	return func(r *RandomTranslation) {
		r.dtype = dtype
	}
}

func RandomTranslationWithTrainable(trainable bool) func(r *RandomTranslation) {
	return func(r *RandomTranslation) {
		r.trainable = trainable
	}
}

func RandomTranslationWithFillMode(fillMode string) func(r *RandomTranslation) {
	return func(r *RandomTranslation) {
		r.fillMode = fillMode
	}
}

func RandomTranslationWithInterpolation(interpolation string) func(r *RandomTranslation) {
	return func(r *RandomTranslation) {
		r.interpolation = interpolation
	}
}

func RandomTranslationWithSeed(seed interface{}) func(r *RandomTranslation) {
	return func(r *RandomTranslation) {
		r.seed = seed
	}
}

func RandomTranslationWithFillValue(fillValue float64) func(r *RandomTranslation) {
	return func(r *RandomTranslation) {
		r.fillValue = fillValue
	}
}

func (r *RandomTranslation) GetShape() tf.Shape {
	return r.shape
}

func (r *RandomTranslation) GetDtype() DataType {
	return r.dtype
}

func (r *RandomTranslation) SetInput(inputs []Layer) {
	r.inputs = inputs
	r.dtype = inputs[0].GetDtype()
}

func (r *RandomTranslation) GetInputs() []Layer {
	return r.inputs
}

func (r *RandomTranslation) GetName() string {
	return r.name
}

type jsonConfigRandomTranslation struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (r *RandomTranslation) GetKerasLayerConfig() interface{} {
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
	return jsonConfigRandomTranslation{
		ClassName: "RandomTranslation",
		Name:      r.name,
		Config: map[string]interface{}{
			"dtype":         r.dtype.String(),
			"fill_mode":     r.fillMode,
			"fill_value":    r.fillValue,
			"height_factor": r.heightFactor,
			"interpolation": r.interpolation,
			"name":          r.name,
			"seed":          r.seed,
			"trainable":     r.trainable,
			"width_factor":  r.widthFactor,
		},
		InboundNodes: inboundNodes,
	}
}

func (r *RandomTranslation) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type RandomZoom struct {
	name          string
	dtype         DataType
	inputs        []Layer
	shape         tf.Shape
	trainable     bool
	heightFactor  float64
	widthFactor   interface{}
	fillMode      string
	interpolation string
	seed          interface{}
	fillValue     float64
}

func NewRandomZoom(heightFactor float64, options ...RandomZoomOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		r := &RandomZoom{
			heightFactor:  heightFactor,
			widthFactor:   nil,
			fillMode:      "reflect",
			interpolation: "bilinear",
			seed:          nil,
			fillValue:     0,
			trainable:     true,
			inputs:        inputs,
			name:          UniqueName("randomzoom"),
		}
		for _, option := range options {
			option(r)
		}
		return r
	}
}

type RandomZoomOption func(*RandomZoom)

func RandomZoomWithName(name string) func(r *RandomZoom) {
	return func(r *RandomZoom) {
		r.name = name
	}
}

func RandomZoomWithDtype(dtype DataType) func(r *RandomZoom) {
	return func(r *RandomZoom) {
		r.dtype = dtype
	}
}

func RandomZoomWithTrainable(trainable bool) func(r *RandomZoom) {
	return func(r *RandomZoom) {
		r.trainable = trainable
	}
}

func RandomZoomWithWidthFactor(widthFactor interface{}) func(r *RandomZoom) {
	return func(r *RandomZoom) {
		r.widthFactor = widthFactor
	}
}

func RandomZoomWithFillMode(fillMode string) func(r *RandomZoom) {
	return func(r *RandomZoom) {
		r.fillMode = fillMode
	}
}

func RandomZoomWithInterpolation(interpolation string) func(r *RandomZoom) {
	return func(r *RandomZoom) {
		r.interpolation = interpolation
	}
}

func RandomZoomWithSeed(seed interface{}) func(r *RandomZoom) {
	return func(r *RandomZoom) {
		r.seed = seed
	}
}

func RandomZoomWithFillValue(fillValue float64) func(r *RandomZoom) {
	return func(r *RandomZoom) {
		r.fillValue = fillValue
	}
}

func (r *RandomZoom) GetShape() tf.Shape {
	return r.shape
}

func (r *RandomZoom) GetDtype() DataType {
	return r.dtype
}

func (r *RandomZoom) SetInput(inputs []Layer) {
	r.inputs = inputs
	r.dtype = inputs[0].GetDtype()
}

func (r *RandomZoom) GetInputs() []Layer {
	return r.inputs
}

func (r *RandomZoom) GetName() string {
	return r.name
}

type jsonConfigRandomZoom struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (r *RandomZoom) GetKerasLayerConfig() interface{} {
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
	return jsonConfigRandomZoom{
		ClassName: "RandomZoom",
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

func (r *RandomZoom) GetCustomLayerDefinition() string {
	return ``
}

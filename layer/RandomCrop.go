package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type RandomCrop struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
	height    float64
	width     float64
	seed      interface{}
}

func NewRandomCrop(height float64, width float64, options ...RandomCropOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		r := &RandomCrop{
			height:    height,
			width:     width,
			seed:      nil,
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("randomcrop"),
		}
		for _, option := range options {
			option(r)
		}
		return r
	}
}

type RandomCropOption func(*RandomCrop)

func RandomCropWithName(name string) func(r *RandomCrop) {
	return func(r *RandomCrop) {
		r.name = name
	}
}

func RandomCropWithDtype(dtype DataType) func(r *RandomCrop) {
	return func(r *RandomCrop) {
		r.dtype = dtype
	}
}

func RandomCropWithTrainable(trainable bool) func(r *RandomCrop) {
	return func(r *RandomCrop) {
		r.trainable = trainable
	}
}

func RandomCropWithSeed(seed interface{}) func(r *RandomCrop) {
	return func(r *RandomCrop) {
		r.seed = seed
	}
}

func (r *RandomCrop) GetShape() tf.Shape {
	return r.shape
}

func (r *RandomCrop) GetDtype() DataType {
	return r.dtype
}

func (r *RandomCrop) SetInput(inputs []Layer) {
	r.inputs = inputs
	r.dtype = inputs[0].GetDtype()
}

func (r *RandomCrop) GetInputs() []Layer {
	return r.inputs
}

func (r *RandomCrop) GetName() string {
	return r.name
}

type jsonConfigRandomCrop struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (r *RandomCrop) GetKerasLayerConfig() interface{} {
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
	return jsonConfigRandomCrop{
		ClassName: "RandomCrop",
		Name:      r.name,
		Config: map[string]interface{}{
			"dtype":     r.dtype.String(),
			"height":    r.height,
			"name":      r.name,
			"seed":      r.seed,
			"trainable": r.trainable,
			"width":     r.width,
		},
		InboundNodes: inboundNodes,
	}
}

func (r *RandomCrop) GetCustomLayerDefinition() string {
	return ``
}

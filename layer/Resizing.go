package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Resizing struct {
	name              string
	dtype             DataType
	inputs            []Layer
	shape             tf.Shape
	trainable         bool
	height            float64
	width             float64
	interpolation     string
	cropToAspectRatio bool
}

func NewResizing(height float64, width float64, options ...ResizingOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		r := &Resizing{
			height:            height,
			width:             width,
			interpolation:     "bilinear",
			cropToAspectRatio: false,
			trainable:         true,
			inputs:            inputs,
			name:              UniqueName("resizing"),
		}
		for _, option := range options {
			option(r)
		}
		return r
	}
}

type ResizingOption func(*Resizing)

func ResizingWithName(name string) func(r *Resizing) {
	return func(r *Resizing) {
		r.name = name
	}
}

func ResizingWithDtype(dtype DataType) func(r *Resizing) {
	return func(r *Resizing) {
		r.dtype = dtype
	}
}

func ResizingWithTrainable(trainable bool) func(r *Resizing) {
	return func(r *Resizing) {
		r.trainable = trainable
	}
}

func ResizingWithInterpolation(interpolation string) func(r *Resizing) {
	return func(r *Resizing) {
		r.interpolation = interpolation
	}
}

func ResizingWithCropToAspectRatio(cropToAspectRatio bool) func(r *Resizing) {
	return func(r *Resizing) {
		r.cropToAspectRatio = cropToAspectRatio
	}
}

func (r *Resizing) GetShape() tf.Shape {
	return r.shape
}

func (r *Resizing) GetDtype() DataType {
	return r.dtype
}

func (r *Resizing) SetInput(inputs []Layer) {
	r.inputs = inputs
	r.dtype = inputs[0].GetDtype()
}

func (r *Resizing) GetInputs() []Layer {
	return r.inputs
}

func (r *Resizing) GetName() string {
	return r.name
}

type jsonConfigResizing struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (r *Resizing) GetKerasLayerConfig() interface{} {
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
	return jsonConfigResizing{
		ClassName: "Resizing",
		Name:      r.name,
		Config: map[string]interface{}{
			"crop_to_aspect_ratio": r.cropToAspectRatio,
			"dtype":                r.dtype.String(),
			"height":               r.height,
			"interpolation":        r.interpolation,
			"name":                 r.name,
			"trainable":            r.trainable,
			"width":                r.width,
		},
		InboundNodes: inboundNodes,
	}
}

func (r *Resizing) GetCustomLayerDefinition() string {
	return ``
}

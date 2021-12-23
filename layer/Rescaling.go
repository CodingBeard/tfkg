package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Rescaling struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
	scale     float64
	offset    float64
}

func NewRescaling(scale float64, options ...RescalingOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		r := &Rescaling{
			scale:     scale,
			offset:    0,
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("rescaling"),
		}
		for _, option := range options {
			option(r)
		}
		return r
	}
}

type RescalingOption func(*Rescaling)

func RescalingWithName(name string) func(r *Rescaling) {
	return func(r *Rescaling) {
		r.name = name
	}
}

func RescalingWithDtype(dtype DataType) func(r *Rescaling) {
	return func(r *Rescaling) {
		r.dtype = dtype
	}
}

func RescalingWithTrainable(trainable bool) func(r *Rescaling) {
	return func(r *Rescaling) {
		r.trainable = trainable
	}
}

func RescalingWithOffset(offset float64) func(r *Rescaling) {
	return func(r *Rescaling) {
		r.offset = offset
	}
}

func (r *Rescaling) GetShape() tf.Shape {
	return r.shape
}

func (r *Rescaling) GetDtype() DataType {
	return r.dtype
}

func (r *Rescaling) SetInput(inputs []Layer) {
	r.inputs = inputs
	r.dtype = inputs[0].GetDtype()
}

func (r *Rescaling) GetInputs() []Layer {
	return r.inputs
}

func (r *Rescaling) GetName() string {
	return r.name
}

type jsonConfigRescaling struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (r *Rescaling) GetKerasLayerConfig() interface{} {
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
	return jsonConfigRescaling{
		ClassName: "Rescaling",
		Name:      r.name,
		Config: map[string]interface{}{
			"dtype":     r.dtype.String(),
			"name":      r.name,
			"offset":    r.offset,
			"scale":     r.scale,
			"trainable": r.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (r *Rescaling) GetCustomLayerDefinition() string {
	return ``
}

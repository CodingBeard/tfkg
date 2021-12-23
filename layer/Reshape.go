package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Reshape struct {
	name        string
	dtype       DataType
	inputs      []Layer
	shape       tf.Shape
	trainable   bool
	targetShape []interface{}
}

func NewReshape(targetShape []interface{}, options ...ReshapeOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		r := &Reshape{
			targetShape: targetShape,
			trainable:   true,
			inputs:      inputs,
			name:        UniqueName("reshape"),
		}
		for _, option := range options {
			option(r)
		}
		return r
	}
}

type ReshapeOption func(*Reshape)

func ReshapeWithName(name string) func(r *Reshape) {
	return func(r *Reshape) {
		r.name = name
	}
}

func ReshapeWithDtype(dtype DataType) func(r *Reshape) {
	return func(r *Reshape) {
		r.dtype = dtype
	}
}

func ReshapeWithTrainable(trainable bool) func(r *Reshape) {
	return func(r *Reshape) {
		r.trainable = trainable
	}
}

func (r *Reshape) GetShape() tf.Shape {
	return r.shape
}

func (r *Reshape) GetDtype() DataType {
	return r.dtype
}

func (r *Reshape) SetInput(inputs []Layer) {
	r.inputs = inputs
	r.dtype = inputs[0].GetDtype()
}

func (r *Reshape) GetInputs() []Layer {
	return r.inputs
}

func (r *Reshape) GetName() string {
	return r.name
}

type jsonConfigReshape struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (r *Reshape) GetKerasLayerConfig() interface{} {
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
	return jsonConfigReshape{
		ClassName: "Reshape",
		Name:      r.name,
		Config: map[string]interface{}{
			"dtype":        r.dtype.String(),
			"name":         r.name,
			"target_shape": r.targetShape,
			"trainable":    r.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (r *Reshape) GetCustomLayerDefinition() string {
	return ``
}

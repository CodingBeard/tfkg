package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Dot struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
	axes      float64
	normalize bool
}

func NewDot(axes float64, options ...DotOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		d := &Dot{
			axes:      axes,
			normalize: false,
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("dot"),
		}
		for _, option := range options {
			option(d)
		}
		return d
	}
}

type DotOption func(*Dot)

func DotWithName(name string) func(d *Dot) {
	return func(d *Dot) {
		d.name = name
	}
}

func DotWithDtype(dtype DataType) func(d *Dot) {
	return func(d *Dot) {
		d.dtype = dtype
	}
}

func DotWithTrainable(trainable bool) func(d *Dot) {
	return func(d *Dot) {
		d.trainable = trainable
	}
}

func DotWithNormalize(normalize bool) func(d *Dot) {
	return func(d *Dot) {
		d.normalize = normalize
	}
}

func (d *Dot) GetShape() tf.Shape {
	return d.shape
}

func (d *Dot) GetDtype() DataType {
	return d.dtype
}

func (d *Dot) SetInput(inputs []Layer) {
	d.inputs = inputs
	d.dtype = inputs[0].GetDtype()
}

func (d *Dot) GetInputs() []Layer {
	return d.inputs
}

func (d *Dot) GetName() string {
	return d.name
}

type jsonConfigDot struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (d *Dot) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range d.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigDot{
		ClassName: "Dot",
		Name:      d.name,
		Config: map[string]interface{}{
			"axes":      d.axes,
			"dtype":     d.dtype.String(),
			"name":      d.name,
			"normalize": d.normalize,
			"trainable": d.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (d *Dot) GetCustomLayerDefinition() string {
	return ``
}

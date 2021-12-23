package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Discretization struct {
	name          string
	dtype         DataType
	inputs        []Layer
	shape         tf.Shape
	trainable     bool
	binBoundaries interface{}
	numBins       interface{}
	epsilon       float64
}

func NewDiscretization(options ...DiscretizationOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		d := &Discretization{
			binBoundaries: nil,
			numBins:       nil,
			epsilon:       0.01,
			trainable:     true,
			inputs:        inputs,
			name:          UniqueName("discretization"),
		}
		for _, option := range options {
			option(d)
		}
		return d
	}
}

type DiscretizationOption func(*Discretization)

func DiscretizationWithName(name string) func(d *Discretization) {
	return func(d *Discretization) {
		d.name = name
	}
}

func DiscretizationWithDtype(dtype DataType) func(d *Discretization) {
	return func(d *Discretization) {
		d.dtype = dtype
	}
}

func DiscretizationWithTrainable(trainable bool) func(d *Discretization) {
	return func(d *Discretization) {
		d.trainable = trainable
	}
}

func DiscretizationWithBinBoundaries(binBoundaries interface{}) func(d *Discretization) {
	return func(d *Discretization) {
		d.binBoundaries = binBoundaries
	}
}

func DiscretizationWithNumBins(numBins interface{}) func(d *Discretization) {
	return func(d *Discretization) {
		d.numBins = numBins
	}
}

func DiscretizationWithEpsilon(epsilon float64) func(d *Discretization) {
	return func(d *Discretization) {
		d.epsilon = epsilon
	}
}

func (d *Discretization) GetShape() tf.Shape {
	return d.shape
}

func (d *Discretization) GetDtype() DataType {
	return d.dtype
}

func (d *Discretization) SetInput(inputs []Layer) {
	d.inputs = inputs
	d.dtype = inputs[0].GetDtype()
}

func (d *Discretization) GetInputs() []Layer {
	return d.inputs
}

func (d *Discretization) GetName() string {
	return d.name
}

type jsonConfigDiscretization struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (d *Discretization) GetKerasLayerConfig() interface{} {
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
	return jsonConfigDiscretization{
		ClassName: "Discretization",
		Name:      d.name,
		Config: map[string]interface{}{
			"bin_boundaries": d.binBoundaries,
			"dtype":          d.dtype.String(),
			"epsilon":        d.epsilon,
			"name":           d.name,
			"num_bins":       d.numBins,
			"trainable":      d.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (d *Discretization) GetCustomLayerDefinition() string {
	return ``
}

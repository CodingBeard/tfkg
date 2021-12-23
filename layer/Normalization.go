package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Normalization struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
	axis      float64
	mean      interface{}
	variance  interface{}
}

func NewNormalization(options ...NormalizationOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		n := &Normalization{
			axis:      -1,
			mean:      nil,
			variance:  nil,
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("normalization"),
		}
		for _, option := range options {
			option(n)
		}
		return n
	}
}

type NormalizationOption func(*Normalization)

func NormalizationWithName(name string) func(n *Normalization) {
	return func(n *Normalization) {
		n.name = name
	}
}

func NormalizationWithDtype(dtype DataType) func(n *Normalization) {
	return func(n *Normalization) {
		n.dtype = dtype
	}
}

func NormalizationWithTrainable(trainable bool) func(n *Normalization) {
	return func(n *Normalization) {
		n.trainable = trainable
	}
}

func NormalizationWithAxis(axis float64) func(n *Normalization) {
	return func(n *Normalization) {
		n.axis = axis
	}
}

func NormalizationWithMean(mean interface{}) func(n *Normalization) {
	return func(n *Normalization) {
		n.mean = mean
	}
}

func NormalizationWithVariance(variance interface{}) func(n *Normalization) {
	return func(n *Normalization) {
		n.variance = variance
	}
}

func (n *Normalization) GetShape() tf.Shape {
	return n.shape
}

func (n *Normalization) GetDtype() DataType {
	return n.dtype
}

func (n *Normalization) SetInput(inputs []Layer) {
	n.inputs = inputs
	n.dtype = inputs[0].GetDtype()
}

func (n *Normalization) GetInputs() []Layer {
	return n.inputs
}

func (n *Normalization) GetName() string {
	return n.name
}

type jsonConfigNormalization struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (n *Normalization) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range n.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigNormalization{
		ClassName: "Normalization",
		Name:      n.name,
		Config: map[string]interface{}{
			"axis":      n.axis,
			"dtype":     n.dtype.String(),
			"mean":      n.mean,
			"name":      n.name,
			"trainable": n.trainable,
			"variance":  n.variance,
		},
		InboundNodes: inboundNodes,
	}
}

func (n *Normalization) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type ActivityRegularization struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
	l1        float64
	l2        float64
}

func NewActivityRegularization(options ...ActivityRegularizationOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		a := &ActivityRegularization{
			l1:        0,
			l2:        0,
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("activityregularization"),
		}
		for _, option := range options {
			option(a)
		}
		return a
	}
}

type ActivityRegularizationOption func(*ActivityRegularization)

func ActivityRegularizationWithName(name string) func(a *ActivityRegularization) {
	return func(a *ActivityRegularization) {
		a.name = name
	}
}

func ActivityRegularizationWithDtype(dtype DataType) func(a *ActivityRegularization) {
	return func(a *ActivityRegularization) {
		a.dtype = dtype
	}
}

func ActivityRegularizationWithTrainable(trainable bool) func(a *ActivityRegularization) {
	return func(a *ActivityRegularization) {
		a.trainable = trainable
	}
}

func ActivityRegularizationWithL1(l1 float64) func(a *ActivityRegularization) {
	return func(a *ActivityRegularization) {
		a.l1 = l1
	}
}

func ActivityRegularizationWithL2(l2 float64) func(a *ActivityRegularization) {
	return func(a *ActivityRegularization) {
		a.l2 = l2
	}
}

func (a *ActivityRegularization) GetShape() tf.Shape {
	return a.shape
}

func (a *ActivityRegularization) GetDtype() DataType {
	return a.dtype
}

func (a *ActivityRegularization) SetInput(inputs []Layer) {
	a.inputs = inputs
	a.dtype = inputs[0].GetDtype()
}

func (a *ActivityRegularization) GetInputs() []Layer {
	return a.inputs
}

func (a *ActivityRegularization) GetName() string {
	return a.name
}

type jsonConfigActivityRegularization struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (a *ActivityRegularization) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range a.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigActivityRegularization{
		ClassName: "ActivityRegularization",
		Name:      a.name,
		Config: map[string]interface{}{
			"dtype":     a.dtype.String(),
			"l1":        a.l1,
			"l2":        a.l2,
			"name":      a.name,
			"trainable": a.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (a *ActivityRegularization) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type TimeDistributed struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
	layer     interface{}
}

func NewTimeDistributed(layer interface{}, options ...TimeDistributedOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		t := &TimeDistributed{
			layer:     layer,
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("timedistributed"),
		}
		for _, option := range options {
			option(t)
		}
		return t
	}
}

type TimeDistributedOption func(*TimeDistributed)

func TimeDistributedWithName(name string) func(t *TimeDistributed) {
	return func(t *TimeDistributed) {
		t.name = name
	}
}

func TimeDistributedWithDtype(dtype DataType) func(t *TimeDistributed) {
	return func(t *TimeDistributed) {
		t.dtype = dtype
	}
}

func TimeDistributedWithTrainable(trainable bool) func(t *TimeDistributed) {
	return func(t *TimeDistributed) {
		t.trainable = trainable
	}
}

func (t *TimeDistributed) GetShape() tf.Shape {
	return t.shape
}

func (t *TimeDistributed) GetDtype() DataType {
	return t.dtype
}

func (t *TimeDistributed) SetInput(inputs []Layer) {
	t.inputs = inputs
	t.dtype = inputs[0].GetDtype()
}

func (t *TimeDistributed) GetInputs() []Layer {
	return t.inputs
}

func (t *TimeDistributed) GetName() string {
	return t.name
}

type jsonConfigTimeDistributed struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (t *TimeDistributed) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range t.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigTimeDistributed{
		ClassName: "TimeDistributed",
		Name:      t.name,
		Config: map[string]interface{}{
			"dtype":     t.dtype.String(),
			"layer":     t.layer,
			"name":      t.name,
			"trainable": t.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (t *TimeDistributed) GetCustomLayerDefinition() string {
	return ``
}

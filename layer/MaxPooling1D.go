package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type MaxPooling1D struct {
	name       string
	dtype      DataType
	inputs     []Layer
	shape      tf.Shape
	trainable  bool
	poolSize   float64
	strides    interface{}
	padding    string
	dataFormat string
}

func NewMaxPooling1D(options ...MaxPooling1DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		m := &MaxPooling1D{
			poolSize:   2,
			strides:    nil,
			padding:    "valid",
			dataFormat: "channels_last",
			trainable:  true,
			inputs:     inputs,
			name:       UniqueName("maxpooling1d"),
		}
		for _, option := range options {
			option(m)
		}
		return m
	}
}

type MaxPooling1DOption func(*MaxPooling1D)

func MaxPooling1DWithName(name string) func(m *MaxPooling1D) {
	return func(m *MaxPooling1D) {
		m.name = name
	}
}

func MaxPooling1DWithDtype(dtype DataType) func(m *MaxPooling1D) {
	return func(m *MaxPooling1D) {
		m.dtype = dtype
	}
}

func MaxPooling1DWithTrainable(trainable bool) func(m *MaxPooling1D) {
	return func(m *MaxPooling1D) {
		m.trainable = trainable
	}
}

func MaxPooling1DWithPoolSize(poolSize float64) func(m *MaxPooling1D) {
	return func(m *MaxPooling1D) {
		m.poolSize = poolSize
	}
}

func MaxPooling1DWithStrides(strides interface{}) func(m *MaxPooling1D) {
	return func(m *MaxPooling1D) {
		m.strides = strides
	}
}

func MaxPooling1DWithPadding(padding string) func(m *MaxPooling1D) {
	return func(m *MaxPooling1D) {
		m.padding = padding
	}
}

func MaxPooling1DWithDataFormat(dataFormat string) func(m *MaxPooling1D) {
	return func(m *MaxPooling1D) {
		m.dataFormat = dataFormat
	}
}

func (m *MaxPooling1D) GetShape() tf.Shape {
	return m.shape
}

func (m *MaxPooling1D) GetDtype() DataType {
	return m.dtype
}

func (m *MaxPooling1D) SetInput(inputs []Layer) {
	m.inputs = inputs
	m.dtype = inputs[0].GetDtype()
}

func (m *MaxPooling1D) GetInputs() []Layer {
	return m.inputs
}

func (m *MaxPooling1D) GetName() string {
	return m.name
}

type jsonConfigMaxPooling1D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (m *MaxPooling1D) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range m.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigMaxPooling1D{
		ClassName: "MaxPooling1D",
		Name:      m.name,
		Config: map[string]interface{}{
			"data_format": m.dataFormat,
			"dtype":       m.dtype.String(),
			"name":        m.name,
			"padding":     m.padding,
			"pool_size":   m.poolSize,
			"strides":     m.strides,
			"trainable":   m.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (m *MaxPooling1D) GetCustomLayerDefinition() string {
	return ``
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type MaxPooling2D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	poolSize []interface {}
	strides interface{}
	padding string
	dataFormat interface{}
}

func NewMaxPooling2D(options ...MaxPooling2DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		m := &MaxPooling2D{
			poolSize: []interface {}{2, 2},
			strides: nil,
			padding: "valid",
			dataFormat: nil,
			trainable: true,
			inputs: inputs,
			name: uniqueName("maxpooling2d"),		
		}
		for _, option := range options {
			option(m)
		}
		return m
	}
}

type MaxPooling2DOption func (*MaxPooling2D)

func MaxPooling2DWithName(name string) func(m *MaxPooling2D) {
	 return func(m *MaxPooling2D) {
		m.name = name
	}
}

func MaxPooling2DWithDtype(dtype DataType) func(m *MaxPooling2D) {
	 return func(m *MaxPooling2D) {
		m.dtype = dtype
	}
}

func MaxPooling2DWithTrainable(trainable bool) func(m *MaxPooling2D) {
	 return func(m *MaxPooling2D) {
		m.trainable = trainable
	}
}

func MaxPooling2DWithPoolSize(poolSize []interface {}) func(m *MaxPooling2D) {
	 return func(m *MaxPooling2D) {
		m.poolSize = poolSize
	}
}

func MaxPooling2DWithStrides(strides interface{}) func(m *MaxPooling2D) {
	 return func(m *MaxPooling2D) {
		m.strides = strides
	}
}

func MaxPooling2DWithPadding(padding string) func(m *MaxPooling2D) {
	 return func(m *MaxPooling2D) {
		m.padding = padding
	}
}

func MaxPooling2DWithDataFormat(dataFormat interface{}) func(m *MaxPooling2D) {
	 return func(m *MaxPooling2D) {
		m.dataFormat = dataFormat
	}
}


func (m *MaxPooling2D) GetShape() tf.Shape {
	return m.shape
}

func (m *MaxPooling2D) GetDtype() DataType {
	return m.dtype
}

func (m *MaxPooling2D) SetInput(inputs []Layer) {
	m.inputs = inputs
	m.dtype = inputs[0].GetDtype()
}

func (m *MaxPooling2D) GetInputs() []Layer {
	return m.inputs
}

func (m *MaxPooling2D) GetName() string {
	return m.name
}


type jsonConfigMaxPooling2D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (m *MaxPooling2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigMaxPooling2D{
		ClassName: "MaxPooling2D",
		Name: m.name,
		Config: map[string]interface{}{
			"name": m.name,
			"trainable": m.trainable,
			"dtype": m.dtype.String(),
			"pool_size": m.poolSize,
			"padding": m.padding,
			"strides": m.strides,
			"data_format": m.dataFormat,
		},
		InboundNodes: inboundNodes,
	}
}
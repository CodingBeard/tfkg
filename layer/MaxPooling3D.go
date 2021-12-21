package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type MaxPooling3D struct {
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

func NewMaxPooling3D(options ...MaxPooling3DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		m := &MaxPooling3D{
			poolSize: []interface {}{2, 2, 2},
			strides: nil,
			padding: "valid",
			dataFormat: nil,
			trainable: true,
			inputs: inputs,
			name: uniqueName("maxpooling3d"),		
		}
		for _, option := range options {
			option(m)
		}
		return m
	}
}

type MaxPooling3DOption func (*MaxPooling3D)

func MaxPooling3DWithName(name string) func(m *MaxPooling3D) {
	 return func(m *MaxPooling3D) {
		m.name = name
	}
}

func MaxPooling3DWithDtype(dtype DataType) func(m *MaxPooling3D) {
	 return func(m *MaxPooling3D) {
		m.dtype = dtype
	}
}

func MaxPooling3DWithTrainable(trainable bool) func(m *MaxPooling3D) {
	 return func(m *MaxPooling3D) {
		m.trainable = trainable
	}
}

func MaxPooling3DWithPoolSize(poolSize []interface {}) func(m *MaxPooling3D) {
	 return func(m *MaxPooling3D) {
		m.poolSize = poolSize
	}
}

func MaxPooling3DWithStrides(strides interface{}) func(m *MaxPooling3D) {
	 return func(m *MaxPooling3D) {
		m.strides = strides
	}
}

func MaxPooling3DWithPadding(padding string) func(m *MaxPooling3D) {
	 return func(m *MaxPooling3D) {
		m.padding = padding
	}
}

func MaxPooling3DWithDataFormat(dataFormat interface{}) func(m *MaxPooling3D) {
	 return func(m *MaxPooling3D) {
		m.dataFormat = dataFormat
	}
}


func (m *MaxPooling3D) GetShape() tf.Shape {
	return m.shape
}

func (m *MaxPooling3D) GetDtype() DataType {
	return m.dtype
}

func (m *MaxPooling3D) SetInput(inputs []Layer) {
	m.inputs = inputs
	m.dtype = inputs[0].GetDtype()
}

func (m *MaxPooling3D) GetInputs() []Layer {
	return m.inputs
}

func (m *MaxPooling3D) GetName() string {
	return m.name
}


type jsonConfigMaxPooling3D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (m *MaxPooling3D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigMaxPooling3D{
		ClassName: "MaxPooling3D",
		Name: m.name,
		Config: map[string]interface{}{
			"data_format": m.dataFormat,
			"name": m.name,
			"trainable": m.trainable,
			"dtype": m.dtype.String(),
			"pool_size": m.poolSize,
			"padding": m.padding,
			"strides": m.strides,
		},
		InboundNodes: inboundNodes,
	}
}
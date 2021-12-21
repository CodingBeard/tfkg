package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type AveragePooling3D struct {
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

func NewAveragePooling3D(options ...AveragePooling3DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		a := &AveragePooling3D{
			poolSize: []interface {}{2, 2, 2},
			strides: nil,
			padding: "valid",
			dataFormat: nil,
			trainable: true,
			inputs: inputs,
			name: uniqueName("averagepooling3d"),		
		}
		for _, option := range options {
			option(a)
		}
		return a
	}
}

type AveragePooling3DOption func (*AveragePooling3D)

func AveragePooling3DWithName(name string) func(a *AveragePooling3D) {
	 return func(a *AveragePooling3D) {
		a.name = name
	}
}

func AveragePooling3DWithDtype(dtype DataType) func(a *AveragePooling3D) {
	 return func(a *AveragePooling3D) {
		a.dtype = dtype
	}
}

func AveragePooling3DWithTrainable(trainable bool) func(a *AveragePooling3D) {
	 return func(a *AveragePooling3D) {
		a.trainable = trainable
	}
}

func AveragePooling3DWithPoolSize(poolSize []interface {}) func(a *AveragePooling3D) {
	 return func(a *AveragePooling3D) {
		a.poolSize = poolSize
	}
}

func AveragePooling3DWithStrides(strides interface{}) func(a *AveragePooling3D) {
	 return func(a *AveragePooling3D) {
		a.strides = strides
	}
}

func AveragePooling3DWithPadding(padding string) func(a *AveragePooling3D) {
	 return func(a *AveragePooling3D) {
		a.padding = padding
	}
}

func AveragePooling3DWithDataFormat(dataFormat interface{}) func(a *AveragePooling3D) {
	 return func(a *AveragePooling3D) {
		a.dataFormat = dataFormat
	}
}


func (a *AveragePooling3D) GetShape() tf.Shape {
	return a.shape
}

func (a *AveragePooling3D) GetDtype() DataType {
	return a.dtype
}

func (a *AveragePooling3D) SetInput(inputs []Layer) {
	a.inputs = inputs
	a.dtype = inputs[0].GetDtype()
}

func (a *AveragePooling3D) GetInputs() []Layer {
	return a.inputs
}

func (a *AveragePooling3D) GetName() string {
	return a.name
}


type jsonConfigAveragePooling3D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (a *AveragePooling3D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigAveragePooling3D{
		ClassName: "AveragePooling3D",
		Name: a.name,
		Config: map[string]interface{}{
			"data_format": a.dataFormat,
			"name": a.name,
			"trainable": a.trainable,
			"dtype": a.dtype.String(),
			"pool_size": a.poolSize,
			"padding": a.padding,
			"strides": a.strides,
		},
		InboundNodes: inboundNodes,
	}
}
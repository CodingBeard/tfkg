package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type AveragePooling2D struct {
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

func NewAveragePooling2D(options ...AveragePooling2DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		a := &AveragePooling2D{
			poolSize: []interface {}{2, 2},
			strides: nil,
			padding: "valid",
			dataFormat: nil,
			trainable: true,
			inputs: inputs,
			name: uniqueName("averagepooling2d"),		
		}
		for _, option := range options {
			option(a)
		}
		return a
	}
}

type AveragePooling2DOption func (*AveragePooling2D)

func AveragePooling2DWithName(name string) func(a *AveragePooling2D) {
	 return func(a *AveragePooling2D) {
		a.name = name
	}
}

func AveragePooling2DWithDtype(dtype DataType) func(a *AveragePooling2D) {
	 return func(a *AveragePooling2D) {
		a.dtype = dtype
	}
}

func AveragePooling2DWithTrainable(trainable bool) func(a *AveragePooling2D) {
	 return func(a *AveragePooling2D) {
		a.trainable = trainable
	}
}

func AveragePooling2DWithPoolSize(poolSize []interface {}) func(a *AveragePooling2D) {
	 return func(a *AveragePooling2D) {
		a.poolSize = poolSize
	}
}

func AveragePooling2DWithStrides(strides interface{}) func(a *AveragePooling2D) {
	 return func(a *AveragePooling2D) {
		a.strides = strides
	}
}

func AveragePooling2DWithPadding(padding string) func(a *AveragePooling2D) {
	 return func(a *AveragePooling2D) {
		a.padding = padding
	}
}

func AveragePooling2DWithDataFormat(dataFormat interface{}) func(a *AveragePooling2D) {
	 return func(a *AveragePooling2D) {
		a.dataFormat = dataFormat
	}
}


func (a *AveragePooling2D) GetShape() tf.Shape {
	return a.shape
}

func (a *AveragePooling2D) GetDtype() DataType {
	return a.dtype
}

func (a *AveragePooling2D) SetInput(inputs []Layer) {
	a.inputs = inputs
	a.dtype = inputs[0].GetDtype()
}

func (a *AveragePooling2D) GetInputs() []Layer {
	return a.inputs
}

func (a *AveragePooling2D) GetName() string {
	return a.name
}


type jsonConfigAveragePooling2D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (a *AveragePooling2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigAveragePooling2D{
		ClassName: "AveragePooling2D",
		Name: a.name,
		Config: map[string]interface{}{
			"dtype": a.dtype.String(),
			"pool_size": a.poolSize,
			"padding": a.padding,
			"strides": a.strides,
			"data_format": a.dataFormat,
			"name": a.name,
			"trainable": a.trainable,
		},
		InboundNodes: inboundNodes,
	}
}
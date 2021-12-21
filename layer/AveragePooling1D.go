package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type AveragePooling1D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	poolSize float64
	strides interface{}
	padding string
	dataFormat string
}

func NewAveragePooling1D(options ...AveragePooling1DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		a := &AveragePooling1D{
			poolSize: 2,
			strides: nil,
			padding: "valid",
			dataFormat: "channels_last",
			trainable: true,
			inputs: inputs,
			name: uniqueName("averagepooling1d"),		
		}
		for _, option := range options {
			option(a)
		}
		return a
	}
}

type AveragePooling1DOption func (*AveragePooling1D)

func AveragePooling1DWithName(name string) func(a *AveragePooling1D) {
	 return func(a *AveragePooling1D) {
		a.name = name
	}
}

func AveragePooling1DWithDtype(dtype DataType) func(a *AveragePooling1D) {
	 return func(a *AveragePooling1D) {
		a.dtype = dtype
	}
}

func AveragePooling1DWithTrainable(trainable bool) func(a *AveragePooling1D) {
	 return func(a *AveragePooling1D) {
		a.trainable = trainable
	}
}

func AveragePooling1DWithPoolSize(poolSize float64) func(a *AveragePooling1D) {
	 return func(a *AveragePooling1D) {
		a.poolSize = poolSize
	}
}

func AveragePooling1DWithStrides(strides interface{}) func(a *AveragePooling1D) {
	 return func(a *AveragePooling1D) {
		a.strides = strides
	}
}

func AveragePooling1DWithPadding(padding string) func(a *AveragePooling1D) {
	 return func(a *AveragePooling1D) {
		a.padding = padding
	}
}

func AveragePooling1DWithDataFormat(dataFormat string) func(a *AveragePooling1D) {
	 return func(a *AveragePooling1D) {
		a.dataFormat = dataFormat
	}
}


func (a *AveragePooling1D) GetShape() tf.Shape {
	return a.shape
}

func (a *AveragePooling1D) GetDtype() DataType {
	return a.dtype
}

func (a *AveragePooling1D) SetInput(inputs []Layer) {
	a.inputs = inputs
	a.dtype = inputs[0].GetDtype()
}

func (a *AveragePooling1D) GetInputs() []Layer {
	return a.inputs
}

func (a *AveragePooling1D) GetName() string {
	return a.name
}


type jsonConfigAveragePooling1D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (a *AveragePooling1D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigAveragePooling1D{
		ClassName: "AveragePooling1D",
		Name: a.name,
		Config: map[string]interface{}{
			"data_format": a.dataFormat,
			"name": a.name,
			"trainable": a.trainable,
			"dtype": a.dtype.String(),
			"strides": a.strides,
			"pool_size": a.poolSize,
			"padding": a.padding,
		},
		InboundNodes: inboundNodes,
	}
}
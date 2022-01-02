package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LZeroPadding1D struct {
	dtype        DataType
	inputs       []Layer
	name         string
	padding      float64
	shape        tf.Shape
	trainable    bool
	layerWeights []*tf.Tensor
}

func ZeroPadding1D() *LZeroPadding1D {
	return &LZeroPadding1D{
		dtype:     Float32,
		name:      UniqueName("zero_padding1d"),
		padding:   1,
		trainable: true,
	}
}

func (l *LZeroPadding1D) SetDtype(dtype DataType) *LZeroPadding1D {
	l.dtype = dtype
	return l
}

func (l *LZeroPadding1D) SetName(name string) *LZeroPadding1D {
	l.name = name
	return l
}

func (l *LZeroPadding1D) SetPadding(padding float64) *LZeroPadding1D {
	l.padding = padding
	return l
}

func (l *LZeroPadding1D) SetShape(shape tf.Shape) *LZeroPadding1D {
	l.shape = shape
	return l
}

func (l *LZeroPadding1D) SetTrainable(trainable bool) *LZeroPadding1D {
	l.trainable = trainable
	return l
}

func (l *LZeroPadding1D) SetLayerWeights(layerWeights []*tf.Tensor) *LZeroPadding1D {
	l.layerWeights = layerWeights
	return l
}

func (l *LZeroPadding1D) GetShape() tf.Shape {
	return l.shape
}

func (l *LZeroPadding1D) GetDtype() DataType {
	return l.dtype
}

func (l *LZeroPadding1D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LZeroPadding1D) GetInputs() []Layer {
	return l.inputs
}

func (l *LZeroPadding1D) GetName() string {
	return l.name
}

func (l *LZeroPadding1D) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLZeroPadding1D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LZeroPadding1D) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range l.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigLZeroPadding1D{
		ClassName: "ZeroPadding1D",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"padding":   l.padding,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LZeroPadding1D) GetCustomLayerDefinition() string {
	return ``
}

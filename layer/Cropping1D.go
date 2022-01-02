package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LCropping1D struct {
	cropping     []interface{}
	dtype        DataType
	inputs       []Layer
	name         string
	shape        tf.Shape
	trainable    bool
	layerWeights []*tf.Tensor
}

func Cropping1D() *LCropping1D {
	return &LCropping1D{
		cropping:  []interface{}{1, 1},
		dtype:     Float32,
		name:      UniqueName("cropping1d"),
		trainable: true,
	}
}

func (l *LCropping1D) SetCropping(cropping []interface{}) *LCropping1D {
	l.cropping = cropping
	return l
}

func (l *LCropping1D) SetDtype(dtype DataType) *LCropping1D {
	l.dtype = dtype
	return l
}

func (l *LCropping1D) SetName(name string) *LCropping1D {
	l.name = name
	return l
}

func (l *LCropping1D) SetShape(shape tf.Shape) *LCropping1D {
	l.shape = shape
	return l
}

func (l *LCropping1D) SetTrainable(trainable bool) *LCropping1D {
	l.trainable = trainable
	return l
}

func (l *LCropping1D) SetLayerWeights(layerWeights []*tf.Tensor) *LCropping1D {
	l.layerWeights = layerWeights
	return l
}

func (l *LCropping1D) GetShape() tf.Shape {
	return l.shape
}

func (l *LCropping1D) GetDtype() DataType {
	return l.dtype
}

func (l *LCropping1D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LCropping1D) GetInputs() []Layer {
	return l.inputs
}

func (l *LCropping1D) GetName() string {
	return l.name
}

func (l *LCropping1D) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLCropping1D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LCropping1D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLCropping1D{
		ClassName: "Cropping1D",
		Name:      l.name,
		Config: map[string]interface{}{
			"cropping":  l.cropping,
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LCropping1D) GetCustomLayerDefinition() string {
	return ``
}

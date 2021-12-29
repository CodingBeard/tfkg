package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LUpSampling1D struct {
	dtype        DataType
	inputs       []Layer
	name         string
	shape        tf.Shape
	size         float64
	trainable    bool
	layerWeights interface{}
}

func UpSampling1D() *LUpSampling1D {
	return &LUpSampling1D{
		dtype:     Float32,
		name:      UniqueName("up_sampling1d"),
		size:      2,
		trainable: true,
	}
}

func (l *LUpSampling1D) SetDtype(dtype DataType) *LUpSampling1D {
	l.dtype = dtype
	return l
}

func (l *LUpSampling1D) SetName(name string) *LUpSampling1D {
	l.name = name
	return l
}

func (l *LUpSampling1D) SetShape(shape tf.Shape) *LUpSampling1D {
	l.shape = shape
	return l
}

func (l *LUpSampling1D) SetSize(size float64) *LUpSampling1D {
	l.size = size
	return l
}

func (l *LUpSampling1D) SetTrainable(trainable bool) *LUpSampling1D {
	l.trainable = trainable
	return l
}

func (l *LUpSampling1D) SetLayerWeights(layerWeights interface{}) *LUpSampling1D {
	l.layerWeights = layerWeights
	return l
}

func (l *LUpSampling1D) GetShape() tf.Shape {
	return l.shape
}

func (l *LUpSampling1D) GetDtype() DataType {
	return l.dtype
}

func (l *LUpSampling1D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LUpSampling1D) GetInputs() []Layer {
	return l.inputs
}

func (l *LUpSampling1D) GetName() string {
	return l.name
}

func (l *LUpSampling1D) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLUpSampling1D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LUpSampling1D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLUpSampling1D{
		ClassName: "UpSampling1D",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"size":      l.size,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LUpSampling1D) GetCustomLayerDefinition() string {
	return ``
}

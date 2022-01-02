package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LRescaling struct {
	dtype        DataType
	inputs       []Layer
	name         string
	offset       float64
	scale        float64
	shape        tf.Shape
	trainable    bool
	layerWeights []*tf.Tensor
}

func Rescaling(scale float64) *LRescaling {
	return &LRescaling{
		dtype:     Float32,
		name:      UniqueName("rescaling"),
		offset:    0,
		scale:     scale,
		trainable: true,
	}
}

func (l *LRescaling) SetDtype(dtype DataType) *LRescaling {
	l.dtype = dtype
	return l
}

func (l *LRescaling) SetName(name string) *LRescaling {
	l.name = name
	return l
}

func (l *LRescaling) SetOffset(offset float64) *LRescaling {
	l.offset = offset
	return l
}

func (l *LRescaling) SetShape(shape tf.Shape) *LRescaling {
	l.shape = shape
	return l
}

func (l *LRescaling) SetTrainable(trainable bool) *LRescaling {
	l.trainable = trainable
	return l
}

func (l *LRescaling) SetLayerWeights(layerWeights []*tf.Tensor) *LRescaling {
	l.layerWeights = layerWeights
	return l
}

func (l *LRescaling) GetShape() tf.Shape {
	return l.shape
}

func (l *LRescaling) GetDtype() DataType {
	return l.dtype
}

func (l *LRescaling) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LRescaling) GetInputs() []Layer {
	return l.inputs
}

func (l *LRescaling) GetName() string {
	return l.name
}

func (l *LRescaling) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLRescaling struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LRescaling) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLRescaling{
		ClassName: "Rescaling",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"name":      l.name,
			"offset":    l.offset,
			"scale":     l.scale,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LRescaling) GetCustomLayerDefinition() string {
	return ``
}

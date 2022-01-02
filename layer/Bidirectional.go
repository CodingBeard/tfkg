package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LBidirectional struct {
	backwardLayer interface{}
	dtype         DataType
	inputs        []Layer
	layer         interface{}
	mergeMode     string
	name          string
	shape         tf.Shape
	trainable     bool
	weights       interface{}
	layerWeights  []*tf.Tensor
}

func Bidirectional(layer interface{}) *LBidirectional {
	return &LBidirectional{
		backwardLayer: nil,
		dtype:         Float32,
		layer:         layer,
		mergeMode:     "concat",
		name:          UniqueName("bidirectional"),
		trainable:     true,
		weights:       nil,
	}
}

func (l *LBidirectional) SetBackwardLayer(backwardLayer interface{}) *LBidirectional {
	l.backwardLayer = backwardLayer
	return l
}

func (l *LBidirectional) SetDtype(dtype DataType) *LBidirectional {
	l.dtype = dtype
	return l
}

func (l *LBidirectional) SetMergeMode(mergeMode string) *LBidirectional {
	l.mergeMode = mergeMode
	return l
}

func (l *LBidirectional) SetName(name string) *LBidirectional {
	l.name = name
	return l
}

func (l *LBidirectional) SetShape(shape tf.Shape) *LBidirectional {
	l.shape = shape
	return l
}

func (l *LBidirectional) SetTrainable(trainable bool) *LBidirectional {
	l.trainable = trainable
	return l
}

func (l *LBidirectional) SetWeights(weights interface{}) *LBidirectional {
	l.weights = weights
	return l
}

func (l *LBidirectional) SetLayerWeights(layerWeights []*tf.Tensor) *LBidirectional {
	l.layerWeights = layerWeights
	return l
}

func (l *LBidirectional) GetShape() tf.Shape {
	return l.shape
}

func (l *LBidirectional) GetDtype() DataType {
	return l.dtype
}

func (l *LBidirectional) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LBidirectional) GetInputs() []Layer {
	return l.inputs
}

func (l *LBidirectional) GetName() string {
	return l.name
}

func (l *LBidirectional) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLBidirectional struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LBidirectional) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLBidirectional{
		ClassName: "Bidirectional",
		Name:      l.name,
		Config: map[string]interface{}{
			"backward_layer": l.backwardLayer,
			"dtype":          l.dtype.String(),
			"layer":          l.layer,
			"merge_mode":     l.mergeMode,
			"name":           l.name,
			"trainable":      l.trainable,
			"weights":        l.weights,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LBidirectional) GetCustomLayerDefinition() string {
	return ``
}

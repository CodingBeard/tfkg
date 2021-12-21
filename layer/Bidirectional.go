package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Bidirectional struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	layer interface{}
	mergeMode string
	weights interface{}
	backwardLayer interface{}
}

func NewBidirectional(layer interface{}, options ...BidirectionalOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		b := &Bidirectional{
			layer: layer,
			mergeMode: "concat",
			weights: nil,
			backwardLayer: nil,
			trainable: true,
			inputs: inputs,
			name: uniqueName("bidirectional"),		
		}
		for _, option := range options {
			option(b)
		}
		return b
	}
}

type BidirectionalOption func (*Bidirectional)

func BidirectionalWithName(name string) func(b *Bidirectional) {
	 return func(b *Bidirectional) {
		b.name = name
	}
}

func BidirectionalWithDtype(dtype DataType) func(b *Bidirectional) {
	 return func(b *Bidirectional) {
		b.dtype = dtype
	}
}

func BidirectionalWithTrainable(trainable bool) func(b *Bidirectional) {
	 return func(b *Bidirectional) {
		b.trainable = trainable
	}
}

func BidirectionalWithMergeMode(mergeMode string) func(b *Bidirectional) {
	 return func(b *Bidirectional) {
		b.mergeMode = mergeMode
	}
}

func BidirectionalWithWeights(weights interface{}) func(b *Bidirectional) {
	 return func(b *Bidirectional) {
		b.weights = weights
	}
}

func BidirectionalWithBackwardLayer(backwardLayer interface{}) func(b *Bidirectional) {
	 return func(b *Bidirectional) {
		b.backwardLayer = backwardLayer
	}
}


func (b *Bidirectional) GetShape() tf.Shape {
	return b.shape
}

func (b *Bidirectional) GetDtype() DataType {
	return b.dtype
}

func (b *Bidirectional) SetInput(inputs []Layer) {
	b.inputs = inputs
	b.dtype = inputs[0].GetDtype()
}

func (b *Bidirectional) GetInputs() []Layer {
	return b.inputs
}

func (b *Bidirectional) GetName() string {
	return b.name
}


type jsonConfigBidirectional struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (b *Bidirectional) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range b.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigBidirectional{
		ClassName: "Bidirectional",
		Name: b.name,
		Config: map[string]interface{}{
			"merge_mode": b.mergeMode,
			"name": b.name,
			"trainable": b.trainable,
			"dtype": b.dtype.String(),
			"layer": b.layer,
		},
		InboundNodes: inboundNodes,
	}
}
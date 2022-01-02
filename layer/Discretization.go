package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LDiscretization struct {
	binBoundaries interface{}
	dtype         DataType
	epsilon       float64
	inputs        []Layer
	name          string
	numBins       interface{}
	shape         tf.Shape
	trainable     bool
	layerWeights  []*tf.Tensor
}

func Discretization() *LDiscretization {
	return &LDiscretization{
		binBoundaries: nil,
		dtype:         Float32,
		epsilon:       0.01,
		name:          UniqueName("discretization"),
		numBins:       nil,
		trainable:     true,
	}
}

func (l *LDiscretization) SetBinBoundaries(binBoundaries interface{}) *LDiscretization {
	l.binBoundaries = binBoundaries
	return l
}

func (l *LDiscretization) SetDtype(dtype DataType) *LDiscretization {
	l.dtype = dtype
	return l
}

func (l *LDiscretization) SetEpsilon(epsilon float64) *LDiscretization {
	l.epsilon = epsilon
	return l
}

func (l *LDiscretization) SetName(name string) *LDiscretization {
	l.name = name
	return l
}

func (l *LDiscretization) SetNumBins(numBins interface{}) *LDiscretization {
	l.numBins = numBins
	return l
}

func (l *LDiscretization) SetShape(shape tf.Shape) *LDiscretization {
	l.shape = shape
	return l
}

func (l *LDiscretization) SetTrainable(trainable bool) *LDiscretization {
	l.trainable = trainable
	return l
}

func (l *LDiscretization) SetLayerWeights(layerWeights []*tf.Tensor) *LDiscretization {
	l.layerWeights = layerWeights
	return l
}

func (l *LDiscretization) GetShape() tf.Shape {
	return l.shape
}

func (l *LDiscretization) GetDtype() DataType {
	return l.dtype
}

func (l *LDiscretization) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LDiscretization) GetInputs() []Layer {
	return l.inputs
}

func (l *LDiscretization) GetName() string {
	return l.name
}

func (l *LDiscretization) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLDiscretization struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LDiscretization) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLDiscretization{
		ClassName: "Discretization",
		Name:      l.name,
		Config: map[string]interface{}{
			"bin_boundaries": l.binBoundaries,
			"dtype":          l.dtype.String(),
			"epsilon":        l.epsilon,
			"name":           l.name,
			"num_bins":       l.numBins,
			"trainable":      l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LDiscretization) GetCustomLayerDefinition() string {
	return ``
}

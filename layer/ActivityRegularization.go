package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LActivityRegularization struct {
	dtype        DataType
	inputs       []Layer
	l1           float64
	l2           float64
	name         string
	shape        tf.Shape
	trainable    bool
	layerWeights []*tf.Tensor
}

func ActivityRegularization() *LActivityRegularization {
	return &LActivityRegularization{
		dtype:     Float32,
		l1:        0,
		l2:        0,
		name:      UniqueName("activity_regularization"),
		trainable: true,
	}
}

func (l *LActivityRegularization) SetDtype(dtype DataType) *LActivityRegularization {
	l.dtype = dtype
	return l
}

func (l *LActivityRegularization) SetL1(l1 float64) *LActivityRegularization {
	l.l1 = l1
	return l
}

func (l *LActivityRegularization) SetL2(l2 float64) *LActivityRegularization {
	l.l2 = l2
	return l
}

func (l *LActivityRegularization) SetName(name string) *LActivityRegularization {
	l.name = name
	return l
}

func (l *LActivityRegularization) SetShape(shape tf.Shape) *LActivityRegularization {
	l.shape = shape
	return l
}

func (l *LActivityRegularization) SetTrainable(trainable bool) *LActivityRegularization {
	l.trainable = trainable
	return l
}

func (l *LActivityRegularization) SetLayerWeights(layerWeights []*tf.Tensor) *LActivityRegularization {
	l.layerWeights = layerWeights
	return l
}

func (l *LActivityRegularization) GetShape() tf.Shape {
	return l.shape
}

func (l *LActivityRegularization) GetDtype() DataType {
	return l.dtype
}

func (l *LActivityRegularization) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LActivityRegularization) GetInputs() []Layer {
	return l.inputs
}

func (l *LActivityRegularization) GetName() string {
	return l.name
}

func (l *LActivityRegularization) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLActivityRegularization struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LActivityRegularization) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLActivityRegularization{
		ClassName: "ActivityRegularization",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":     l.dtype.String(),
			"l1":        l.l1,
			"l2":        l.l2,
			"name":      l.name,
			"trainable": l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LActivityRegularization) GetCustomLayerDefinition() string {
	return ``
}

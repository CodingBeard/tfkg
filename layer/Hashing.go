package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LHashing struct {
	dtype        DataType
	inputs       []Layer
	maskValue    interface{}
	name         string
	numBins      float64
	salt         interface{}
	shape        tf.Shape
	trainable    bool
	layerWeights interface{}
}

func Hashing(numBins float64) *LHashing {
	return &LHashing{
		dtype:     Float32,
		maskValue: nil,
		name:      UniqueName("hashing"),
		numBins:   numBins,
		salt:      nil,
		trainable: true,
	}
}

func (l *LHashing) SetDtype(dtype DataType) *LHashing {
	l.dtype = dtype
	return l
}

func (l *LHashing) SetMaskValue(maskValue interface{}) *LHashing {
	l.maskValue = maskValue
	return l
}

func (l *LHashing) SetName(name string) *LHashing {
	l.name = name
	return l
}

func (l *LHashing) SetSalt(salt interface{}) *LHashing {
	l.salt = salt
	return l
}

func (l *LHashing) SetShape(shape tf.Shape) *LHashing {
	l.shape = shape
	return l
}

func (l *LHashing) SetTrainable(trainable bool) *LHashing {
	l.trainable = trainable
	return l
}

func (l *LHashing) SetLayerWeights(layerWeights interface{}) *LHashing {
	l.layerWeights = layerWeights
	return l
}

func (l *LHashing) GetShape() tf.Shape {
	return l.shape
}

func (l *LHashing) GetDtype() DataType {
	return l.dtype
}

func (l *LHashing) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LHashing) GetInputs() []Layer {
	return l.inputs
}

func (l *LHashing) GetName() string {
	return l.name
}

func (l *LHashing) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLHashing struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LHashing) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLHashing{
		ClassName: "Hashing",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":      l.dtype.String(),
			"mask_value": l.maskValue,
			"name":       l.name,
			"num_bins":   l.numBins,
			"salt":       l.salt,
			"trainable":  l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LHashing) GetCustomLayerDefinition() string {
	return ``
}

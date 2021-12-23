package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Hashing struct {
	name      string
	dtype     DataType
	inputs    []Layer
	shape     tf.Shape
	trainable bool
	numBins   float64
	maskValue interface{}
	salt      interface{}
}

func NewHashing(numBins float64, options ...HashingOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		h := &Hashing{
			numBins:   numBins,
			maskValue: nil,
			salt:      nil,
			trainable: true,
			inputs:    inputs,
			name:      UniqueName("hashing"),
		}
		for _, option := range options {
			option(h)
		}
		return h
	}
}

type HashingOption func(*Hashing)

func HashingWithName(name string) func(h *Hashing) {
	return func(h *Hashing) {
		h.name = name
	}
}

func HashingWithDtype(dtype DataType) func(h *Hashing) {
	return func(h *Hashing) {
		h.dtype = dtype
	}
}

func HashingWithTrainable(trainable bool) func(h *Hashing) {
	return func(h *Hashing) {
		h.trainable = trainable
	}
}

func HashingWithMaskValue(maskValue interface{}) func(h *Hashing) {
	return func(h *Hashing) {
		h.maskValue = maskValue
	}
}

func HashingWithSalt(salt interface{}) func(h *Hashing) {
	return func(h *Hashing) {
		h.salt = salt
	}
}

func (h *Hashing) GetShape() tf.Shape {
	return h.shape
}

func (h *Hashing) GetDtype() DataType {
	return h.dtype
}

func (h *Hashing) SetInput(inputs []Layer) {
	h.inputs = inputs
	h.dtype = inputs[0].GetDtype()
}

func (h *Hashing) GetInputs() []Layer {
	return h.inputs
}

func (h *Hashing) GetName() string {
	return h.name
}

type jsonConfigHashing struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (h *Hashing) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range h.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigHashing{
		ClassName: "Hashing",
		Name:      h.name,
		Config: map[string]interface{}{
			"dtype":      h.dtype.String(),
			"mask_value": h.maskValue,
			"name":       h.name,
			"num_bins":   h.numBins,
			"salt":       h.salt,
			"trainable":  h.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (h *Hashing) GetCustomLayerDefinition() string {
	return ``
}

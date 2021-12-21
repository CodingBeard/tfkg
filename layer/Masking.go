package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Masking struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	maskValue float64
}

func NewMasking(options ...MaskingOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		m := &Masking{
			maskValue: 0,
			trainable: true,
			inputs: inputs,
			name: uniqueName("masking"),		
		}
		for _, option := range options {
			option(m)
		}
		return m
	}
}

type MaskingOption func (*Masking)

func MaskingWithName(name string) func(m *Masking) {
	 return func(m *Masking) {
		m.name = name
	}
}

func MaskingWithDtype(dtype DataType) func(m *Masking) {
	 return func(m *Masking) {
		m.dtype = dtype
	}
}

func MaskingWithTrainable(trainable bool) func(m *Masking) {
	 return func(m *Masking) {
		m.trainable = trainable
	}
}

func MaskingWithMaskValue(maskValue float64) func(m *Masking) {
	 return func(m *Masking) {
		m.maskValue = maskValue
	}
}


func (m *Masking) GetShape() tf.Shape {
	return m.shape
}

func (m *Masking) GetDtype() DataType {
	return m.dtype
}

func (m *Masking) SetInput(inputs []Layer) {
	m.inputs = inputs
	m.dtype = inputs[0].GetDtype()
}

func (m *Masking) GetInputs() []Layer {
	return m.inputs
}

func (m *Masking) GetName() string {
	return m.name
}


type jsonConfigMasking struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (m *Masking) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range m.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigMasking{
		ClassName: "Masking",
		Name: m.name,
		Config: map[string]interface{}{
			"name": m.name,
			"trainable": m.trainable,
			"dtype": m.dtype.String(),
			"mask_value": m.maskValue,
		},
		InboundNodes: inboundNodes,
	}
}
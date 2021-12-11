package layer

import (
	tf "github.com/galeone/tensorflow/tensorflow/go"
)

type Dropout struct {
	rate   float32
	name   string
	dtype  DataType
	inputs []Layer
}

type DropoutConfig struct {
	Name string
}

func NewDropout(rate float32, optionalConfig ...DropoutConfig) func(inputs ...Layer) Layer {
	var config DropoutConfig
	if len(optionalConfig) == 1 {
		config = optionalConfig[0]
	}

	if config.Name == "" {
		config.Name = uniqueName("dropout")
	}

	return func(inputs ...Layer) Layer {
		return &Dropout{
			rate:   rate,
			name:   config.Name,
			inputs: inputs,
		}
	}

}

func (b *Dropout) GetShape() tf.Shape {
	return tf.MakeShape()
}

func (b *Dropout) GetDtype() DataType {
	return b.dtype
}

func (b *Dropout) SetInput(inputs []Layer) {
	b.inputs = inputs
	b.dtype = inputs[0].GetDtype()
}

func (b *Dropout) GetInputs() []Layer {
	return b.inputs
}

func (b *Dropout) GetName() string {
	return b.name
}

type kerasDropoutConfig struct {
	ClassName string `json:"class_name"`
	Config    struct {
		Name       string      `json:"name"`
		Trainable  bool        `json:"trainable"`
		Dtype      string      `json:"dtype"`
		Rate       float32     `json:"rate"`
		NoiseShape interface{} `json:"noise_shape"`
		Seed       interface{} `json:"seed"`
	} `json:"config"`
	Name         string            `json:"name"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}

func (b *Dropout) GetKerasLayerConfig() interface{} {
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
	config := kerasDropoutConfig{
		ClassName: "Dropout",
		Config: struct {
			Name       string      `json:"name"`
			Trainable  bool        `json:"trainable"`
			Dtype      string      `json:"dtype"`
			Rate       float32     `json:"rate"`
			NoiseShape interface{} `json:"noise_shape"`
			Seed       interface{} `json:"seed"`
		}{
			Name:       b.name,
			Trainable:  true,
			Dtype:      string(b.dtype),
			Rate:       b.rate,
			NoiseShape: nil,
			Seed:       nil,
		},
		Name:         b.name,
		InboundNodes: inboundNodes,
	}

	return config
}

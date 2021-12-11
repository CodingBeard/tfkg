package layer

import (
	tf "github.com/galeone/tensorflow/tensorflow/go"
)

type Dense struct {
	units      int
	shape      tf.Shape
	name       string
	dtype      DataType
	activation string
	inputs     []Layer
}

type DenseConfig struct {
	Name       string
	Activation string
}

func NewDense(units int, dType DataType, optionalConfig ...DenseConfig) func(inputs ...Layer) Layer {
	var config DenseConfig
	if len(optionalConfig) == 1 {
		config = optionalConfig[0]
	}

	if config.Name == "" {
		config.Name = uniqueName("dense")
	}

	if config.Activation == "" {
		config.Activation = "linear"
	}

	return func(inputs ...Layer) Layer {
		return &Dense{
			units:      units,
			shape:      tf.MakeShape(-1, int64(units)),
			name:       config.Name,
			dtype:      dType,
			activation: config.Activation,
			inputs:     inputs,
		}
	}
}

func (d *Dense) GetShape() tf.Shape {
	return d.shape
}

func (d *Dense) GetDtype() DataType {
	return d.dtype
}

func (d *Dense) SetInput(inputs []Layer) {
	d.inputs = inputs
	d.dtype = inputs[0].GetDtype()
}

func (d *Dense) GetInputs() []Layer {
	return d.inputs
}

func (d *Dense) GetName() string {
	return d.name
}

type kerasDenseConfig struct {
	ClassName string `json:"class_name"`
	Config    struct {
		Name                string                            `json:"name"`
		Trainable           bool                              `json:"trainable"`
		Dtype               string                            `json:"dtype"`
		Units               int                               `json:"units"`
		Activation          string                            `json:"activation"`
		UseBias             bool                              `json:"use_bias"`
		KernelInitializer   kerasDenseKernelInitializerConfig `json:"kernel_initializer"`
		BiasInitializer     kerasDenseBiasInitializer         `json:"bias_initializer"`
		KernelRegularizer   interface{}                       `json:"kernel_regularizer"`
		BiasRegularizer     interface{}                       `json:"bias_regularizer"`
		ActivityRegularizer interface{}                       `json:"activity_regularizer"`
		KernelConstraint    interface{}                       `json:"kernel_constraint"`
		BiasConstraint      interface{}                       `json:"bias_constraint"`
	} `json:"config"`
	Name         string            `json:"name"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}

type kerasDenseKernelInitializerConfig struct {
	ClassName string `json:"class_name"`
	Config    struct {
		Seed interface{} `json:"seed"`
	} `json:"config"`
}

type kerasDenseBiasInitializer struct {
	ClassName string   `json:"class_name"`
	Config    struct{} `json:"config"`
}

func (d *Dense) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range d.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	config := kerasDenseConfig{
		ClassName: "Dense",
		Config: struct {
			Name                string                            `json:"name"`
			Trainable           bool                              `json:"trainable"`
			Dtype               string                            `json:"dtype"`
			Units               int                               `json:"units"`
			Activation          string                            `json:"activation"`
			UseBias             bool                              `json:"use_bias"`
			KernelInitializer   kerasDenseKernelInitializerConfig `json:"kernel_initializer"`
			BiasInitializer     kerasDenseBiasInitializer         `json:"bias_initializer"`
			KernelRegularizer   interface{}                       `json:"kernel_regularizer"`
			BiasRegularizer     interface{}                       `json:"bias_regularizer"`
			ActivityRegularizer interface{}                       `json:"activity_regularizer"`
			KernelConstraint    interface{}                       `json:"kernel_constraint"`
			BiasConstraint      interface{}                       `json:"bias_constraint"`
		}{
			Name:       d.name,
			Trainable:  true,
			Dtype:      string(d.dtype),
			Units:      d.units,
			Activation: d.activation,
			UseBias:    true,
			KernelInitializer: kerasDenseKernelInitializerConfig{
				ClassName: "GlorotUniform",
				Config: struct {
					Seed interface{} `json:"seed"`
				}{
					Seed: nil,
				},
			},
			BiasInitializer: kerasDenseBiasInitializer{
				ClassName: "Zeros",
				Config:    struct{}{},
			},
			KernelRegularizer:   nil,
			BiasRegularizer:     nil,
			ActivityRegularizer: nil,
			KernelConstraint:    nil,
			BiasConstraint:      nil,
		},
		Name:         d.name,
		InboundNodes: inboundNodes,
	}

	return config
}

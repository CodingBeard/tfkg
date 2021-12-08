package layer

import (
	tf "github.com/galeone/tensorflow/tensorflow/go"
)

type Dense struct {
	units             int
	shape             tf.Shape
	name              string
	dtype             DataType
	activation        string
	useBias           TfBool
	kernelInitializer string
	biasInitializer   string
	input             Layer
}

type DenseConfig struct {
	Name              string
	Activation        string
	UseBias           TfBool
	KernelInitializer string
	BiasInitializer   string
}

func NewDense(units int, dType DataType, optionalConfig ...DenseConfig) func(inputs ...Layer) Layer {
	var config DenseConfig
	if len(optionalConfig) == 1 {
		config = optionalConfig[0]
	}

	if config.Name == "" {
		config.Name = uniqueName("dense")
	}

	if config.UseBias == "" {
		config.UseBias = TfTrue
	}

	if config.KernelInitializer == "" {
		config.KernelInitializer = "glorot_uniform"
	}

	if config.BiasInitializer == "" {
		config.BiasInitializer = "zeros"
	}

	return func(inputs ...Layer) Layer {
		return &Dense{
			units:             units,
			shape:             tf.MakeShape(-1, int64(units)),
			name:              config.Name,
			dtype:             dType,
			useBias:           config.UseBias,
			kernelInitializer: config.KernelInitializer,
			biasInitializer:   config.BiasInitializer,
			activation:        config.Activation,
			input:             inputs[0],
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
	d.input = inputs[0]
}

func (d *Dense) GetInputs() []Layer {
	return []Layer{d.input}
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
			UseBias:    d.useBias.ToBool(true),
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
		Name: d.name,
		InboundNodes: [][][]interface{}{
			{
				{
					d.input.GetName(),
					0,
					0,
					map[string]bool{},
				},
			},
		},
	}

	//TODO implement kernalInitializer
	if d.kernelInitializer != "glorot_uniform" {
		panic("Dense.kernalInitializer not implemented")
	}

	//TODO implement biasInitializer
	if d.biasInitializer != "zeros" {
		panic("Dense.biasInitializer not implemented")
	}

	return config
}

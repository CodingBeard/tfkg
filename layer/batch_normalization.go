package layer

import (
	tf "github.com/galeone/tensorflow/tensorflow/go"
)

type BatchNormalization struct {
	axis   int
	name   string
	dtype  DataType
	inputs []Layer
}

type BatchNormalizationConfig struct {
	Name string
}

func NewBatchNormalization(axis int, optionalConfig ...BatchNormalizationConfig) func(inputs ...Layer) Layer {
	var config BatchNormalizationConfig
	if len(optionalConfig) == 1 {
		config = optionalConfig[0]
	}

	if config.Name == "" {
		config.Name = uniqueName("batch_normalization")
	}

	return func(inputs ...Layer) Layer {
		return &BatchNormalization{
			axis:   axis,
			name:   config.Name,
			inputs: inputs,
		}
	}

}

func (b *BatchNormalization) GetShape() tf.Shape {
	return tf.MakeShape()
}

func (b *BatchNormalization) GetDtype() DataType {
	return b.dtype
}

func (b *BatchNormalization) SetInput(inputs []Layer) {
	b.inputs = inputs
	b.dtype = inputs[0].GetDtype()
}

func (b *BatchNormalization) GetInputs() []Layer {
	return b.inputs
}

func (b *BatchNormalization) GetName() string {
	return b.name
}

type kerasBatchNormalizationConfig struct {
	ClassName string `json:"class_name"`
	Config    struct {
		Name            string  `json:"name"`
		Trainable       bool    `json:"trainable"`
		Dtype           string  `json:"dtype"`
		Axis            []int   `json:"axis"`
		Momentum        float64 `json:"momentum"`
		Epsilon         float64 `json:"epsilon"`
		Center          bool    `json:"center"`
		Scale           bool    `json:"scale"`
		BetaInitializer struct {
			ClassName string `json:"class_name"`
			Config    struct {
			} `json:"config"`
		} `json:"beta_initializer"`
		GammaInitializer struct {
			ClassName string `json:"class_name"`
			Config    struct {
			} `json:"config"`
		} `json:"gamma_initializer"`
		MovingMeanInitializer struct {
			ClassName string `json:"class_name"`
			Config    struct {
			} `json:"config"`
		} `json:"moving_mean_initializer"`
		MovingVarianceInitializer struct {
			ClassName string `json:"class_name"`
			Config    struct {
			} `json:"config"`
		} `json:"moving_variance_initializer"`
		BetaRegularizer  interface{} `json:"beta_regularizer"`
		GammaRegularizer interface{} `json:"gamma_regularizer"`
		BetaConstraint   interface{} `json:"beta_constraint"`
		GammaConstraint  interface{} `json:"gamma_constraint"`
	} `json:"config"`
	Name         string            `json:"name"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}

func (b *BatchNormalization) GetKerasLayerConfig() interface{} {
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
	config := kerasBatchNormalizationConfig{
		ClassName: "BatchNormalization",
		Config: struct {
			Name            string  `json:"name"`
			Trainable       bool    `json:"trainable"`
			Dtype           string  `json:"dtype"`
			Axis            []int   `json:"axis"`
			Momentum        float64 `json:"momentum"`
			Epsilon         float64 `json:"epsilon"`
			Center          bool    `json:"center"`
			Scale           bool    `json:"scale"`
			BetaInitializer struct {
				ClassName string   `json:"class_name"`
				Config    struct{} `json:"config"`
			} `json:"beta_initializer"`
			GammaInitializer struct {
				ClassName string   `json:"class_name"`
				Config    struct{} `json:"config"`
			} `json:"gamma_initializer"`
			MovingMeanInitializer struct {
				ClassName string   `json:"class_name"`
				Config    struct{} `json:"config"`
			} `json:"moving_mean_initializer"`
			MovingVarianceInitializer struct {
				ClassName string   `json:"class_name"`
				Config    struct{} `json:"config"`
			} `json:"moving_variance_initializer"`
			BetaRegularizer  interface{} `json:"beta_regularizer"`
			GammaRegularizer interface{} `json:"gamma_regularizer"`
			BetaConstraint   interface{} `json:"beta_constraint"`
			GammaConstraint  interface{} `json:"gamma_constraint"`
		}{
			Name:      b.name,
			Trainable: true,
			Dtype:     string(b.dtype),
			Axis:      []int{b.axis},
			Momentum:  0.99,
			Epsilon:   0.001,
			Center:    true,
			Scale:     true,
			BetaInitializer: struct {
				ClassName string   `json:"class_name"`
				Config    struct{} `json:"config"`
			}{
				ClassName: "Zeros",
				Config:    struct{}{},
			},
			GammaInitializer: struct {
				ClassName string   `json:"class_name"`
				Config    struct{} `json:"config"`
			}{
				ClassName: "Ones",
				Config:    struct{}{},
			},
			MovingMeanInitializer: struct {
				ClassName string   `json:"class_name"`
				Config    struct{} `json:"config"`
			}{
				ClassName: "Zeros",
				Config:    struct{}{},
			},
			MovingVarianceInitializer: struct {
				ClassName string   `json:"class_name"`
				Config    struct{} `json:"config"`
			}{
				ClassName: "Ones",
				Config:    struct{}{},
			},
			BetaRegularizer:  nil,
			GammaRegularizer: nil,
			BetaConstraint:   nil,
			GammaConstraint:  nil,
		},
		Name:         b.name,
		InboundNodes: inboundNodes,
	}

	return config
}

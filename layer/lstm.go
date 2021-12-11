package layer

import (
	tf "github.com/galeone/tensorflow/tensorflow/go"
)

type LSTM struct {
	units  int
	name   string
	dtype  DataType
	inputs []Layer
}

type LSTMConfig struct {
	Name string
}

func NewLSTM(units int, optionalConfig ...LSTMConfig) func(inputs ...Layer) Layer {
	var config LSTMConfig
	if len(optionalConfig) == 1 {
		config = optionalConfig[0]
	}

	if config.Name == "" {
		config.Name = uniqueName("lstm")
	}

	return func(inputs ...Layer) Layer {
		return &LSTM{
			units:  units,
			name:   config.Name,
			inputs: inputs,
		}
	}

}

func (l *LSTM) GetShape() tf.Shape {
	return tf.MakeShape()
}

func (l *LSTM) GetDtype() DataType {
	return l.dtype
}

func (l *LSTM) SetInput(inputs []Layer) {
	l.inputs = inputs
	l.dtype = inputs[0].GetDtype()
}

func (l *LSTM) GetInputs() []Layer {
	return l.inputs
}

func (l *LSTM) GetName() string {
	return l.name
}

type kerasLstmConfig struct {
	ClassName string `json:"class_name"`
	Config    struct {
		Name                string `json:"name"`
		Trainable           bool   `json:"trainable"`
		Dtype               string `json:"dtype"`
		ReturnSequences     bool   `json:"return_sequences"`
		ReturnState         bool   `json:"return_state"`
		GoBackwards         bool   `json:"go_backwards"`
		Stateful            bool   `json:"stateful"`
		Unroll              bool   `json:"unroll"`
		TimeMajor           bool   `json:"time_major"`
		Units               int    `json:"units"`
		Activation          string `json:"activation"`
		RecurrentActivation string `json:"recurrent_activation"`
		UseBias             bool   `json:"use_bias"`
		KernelInitializer   struct {
			ClassName string `json:"class_name"`
			Config    struct {
				Seed interface{} `json:"seed"`
			} `json:"config"`
			SharedObjectID int `json:"shared_object_id"`
		} `json:"kernel_initializer"`
		RecurrentInitializer struct {
			ClassName string `json:"class_name"`
			Config    struct {
				Gain int         `json:"gain"`
				Seed interface{} `json:"seed"`
			} `json:"config"`
			SharedObjectID int `json:"shared_object_id"`
		} `json:"recurrent_initializer"`
		BiasInitializer struct {
			ClassName string `json:"class_name"`
			Config    struct {
			} `json:"config"`
			SharedObjectID int `json:"shared_object_id"`
		} `json:"bias_initializer"`
		UnitForgetBias       bool        `json:"unit_forget_bias"`
		KernelRegularizer    interface{} `json:"kernel_regularizer"`
		RecurrentRegularizer interface{} `json:"recurrent_regularizer"`
		BiasRegularizer      interface{} `json:"bias_regularizer"`
		ActivityRegularizer  interface{} `json:"activity_regularizer"`
		KernelConstraint     interface{} `json:"kernel_constraint"`
		RecurrentConstraint  interface{} `json:"recurrent_constraint"`
		BiasConstraint       interface{} `json:"bias_constraint"`
		Dropout              int         `json:"dropout"`
		RecurrentDropout     int         `json:"recurrent_dropout"`
		Implementation       int         `json:"implementation"`
	} `json:"config"`
	Name         string            `json:"name"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}

func (l *LSTM) GetKerasLayerConfig() interface{} {
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
	config := kerasLstmConfig{
		ClassName: "LSTM",
		Config: struct {
			Name                string `json:"name"`
			Trainable           bool   `json:"trainable"`
			Dtype               string `json:"dtype"`
			ReturnSequences     bool   `json:"return_sequences"`
			ReturnState         bool   `json:"return_state"`
			GoBackwards         bool   `json:"go_backwards"`
			Stateful            bool   `json:"stateful"`
			Unroll              bool   `json:"unroll"`
			TimeMajor           bool   `json:"time_major"`
			Units               int    `json:"units"`
			Activation          string `json:"activation"`
			RecurrentActivation string `json:"recurrent_activation"`
			UseBias             bool   `json:"use_bias"`
			KernelInitializer   struct {
				ClassName string `json:"class_name"`
				Config    struct {
					Seed interface{} `json:"seed"`
				} `json:"config"`
				SharedObjectID int `json:"shared_object_id"`
			} `json:"kernel_initializer"`
			RecurrentInitializer struct {
				ClassName string `json:"class_name"`
				Config    struct {
					Gain int         `json:"gain"`
					Seed interface{} `json:"seed"`
				} `json:"config"`
				SharedObjectID int `json:"shared_object_id"`
			} `json:"recurrent_initializer"`
			BiasInitializer struct {
				ClassName      string   `json:"class_name"`
				Config         struct{} `json:"config"`
				SharedObjectID int      `json:"shared_object_id"`
			} `json:"bias_initializer"`
			UnitForgetBias       bool        `json:"unit_forget_bias"`
			KernelRegularizer    interface{} `json:"kernel_regularizer"`
			RecurrentRegularizer interface{} `json:"recurrent_regularizer"`
			BiasRegularizer      interface{} `json:"bias_regularizer"`
			ActivityRegularizer  interface{} `json:"activity_regularizer"`
			KernelConstraint     interface{} `json:"kernel_constraint"`
			RecurrentConstraint  interface{} `json:"recurrent_constraint"`
			BiasConstraint       interface{} `json:"bias_constraint"`
			Dropout              int         `json:"dropout"`
			RecurrentDropout     int         `json:"recurrent_dropout"`
			Implementation       int         `json:"implementation"`
		}{
			Name:                l.name,
			Trainable:           true,
			Dtype:               string(l.dtype),
			ReturnSequences:     false,
			ReturnState:         false,
			GoBackwards:         false,
			Stateful:            false,
			Unroll:              false,
			TimeMajor:           false,
			Units:               l.units,
			Activation:          "tanh",
			RecurrentActivation: "sigmoid",
			UseBias:             true,
			KernelInitializer: struct {
				ClassName string `json:"class_name"`
				Config    struct {
					Seed interface{} `json:"seed"`
				} `json:"config"`
				SharedObjectID int `json:"shared_object_id"`
			}{
				ClassName: "GlorotUniform",
				Config: struct {
					Seed interface{} `json:"seed"`
				}{},
				SharedObjectID: 3, // TODO: where are this generated from?
			},
			RecurrentInitializer: struct {
				ClassName string `json:"class_name"`
				Config    struct {
					Gain int         `json:"gain"`
					Seed interface{} `json:"seed"`
				} `json:"config"`
				SharedObjectID int `json:"shared_object_id"`
			}{
				ClassName: "Orthogonal",
				Config: struct {
					Gain int         `json:"gain"`
					Seed interface{} `json:"seed"`
				}{
					Gain: 1,
					Seed: nil,
				},
				SharedObjectID: 4, // TODO: where are this generated from?
			},
			BiasInitializer: struct {
				ClassName      string   `json:"class_name"`
				Config         struct{} `json:"config"`
				SharedObjectID int      `json:"shared_object_id"`
			}{
				ClassName:      "Zeros",
				Config:         struct{}{},
				SharedObjectID: 5, // TODO: where are this generated from?
			},
			UnitForgetBias:       true,
			KernelRegularizer:    nil,
			RecurrentRegularizer: nil,
			BiasRegularizer:      nil,
			ActivityRegularizer:  nil,
			KernelConstraint:     nil,
			RecurrentConstraint:  nil,
			BiasConstraint:       nil,
			Dropout:              0,
			RecurrentDropout:     0,
			Implementation:       2,
		},
		Name:         l.name,
		InboundNodes: inboundNodes,
	}

	return config
}

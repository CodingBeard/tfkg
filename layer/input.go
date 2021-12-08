package layer

import (
	tf "github.com/galeone/tensorflow/tensorflow/go"
)

type Input struct {
	shape     tf.Shape
	batchSize int
	name      string
	dtype     DataType
	sparse    TfBool
	ragged    TfBool
}

type InputConfig struct {
	BatchSize int
	Name      string
	Sparse    TfBool
	Ragged    TfBool
}

func NewInput(shape tf.Shape, dType DataType, optionalConfig ...InputConfig) *Input {
	var config InputConfig
	if len(optionalConfig) == 1 {
		config = optionalConfig[0]
	}

	if config.Name == "" {
		config.Name = uniqueName("input")
	}
	if config.Sparse == "" {
		config.Sparse = TfDefault
	}

	if config.Ragged == "" {
		config.Ragged = TfDefault
	}

	return &Input{
		shape:     shape,
		batchSize: config.BatchSize,
		name:      config.Name,
		dtype:     dType,
		sparse:    config.Sparse,
		ragged:    config.Ragged,
	}
}

func (i *Input) GetShape() tf.Shape {
	return i.shape
}

func (i *Input) GetDtype() DataType {
	return i.dtype
}

func (i *Input) SetInput(inputs []Layer) {

}

func (i *Input) GetInputs() []Layer {
	return []Layer{}
}

func (i *Input) GetName() string {
	return i.name
}

type kerasInputConfig struct {
	ClassName string `json:"class_name"`
	Config    struct {
		BatchInputShape []interface{} `json:"batch_input_shape"`
		Dtype           string        `json:"dtype"`
		Sparse          bool          `json:"sparse"`
		Ragged          bool          `json:"ragged"`
		Name            string        `json:"name"`
	} `json:"config"`
	Name         string        `json:"name"`
	InboundNodes []interface{} `json:"inbound_nodes"`
}

func (i *Input) GetKerasLayerConfig() interface{} {
	shape := []interface{}{
		nil,
		nil,
	}

	dims, _ := i.shape.ToSlice()

	for _, dim := range dims {
		if dim == -1 {
			shape = append(shape, nil)
		} else {
			shape = append(shape, dim)
		}
	}

	config := kerasInputConfig{
		ClassName: "InputLayer",
		Config: struct {
			BatchInputShape []interface{} `json:"batch_input_shape"`
			Dtype           string        `json:"dtype"`
			Sparse          bool          `json:"sparse"`
			Ragged          bool          `json:"ragged"`
			Name            string        `json:"name"`
		}{
			BatchInputShape: shape,
			Dtype:           string(i.dtype),
			Sparse:          i.sparse.ToBool(false),
			Ragged:          i.ragged.ToBool(false),
			Name:            i.name,
		},
		Name:         i.name,
		InboundNodes: []interface{}{},
	}

	return config
}

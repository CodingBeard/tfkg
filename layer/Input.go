package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Input struct {
	name        string
	dtype       DataType
	shape       tf.Shape
	trainable   bool
	batchSize   float64
	inputTensor interface{}
	sparse      bool
	ragged      bool
}

func NewInput(options ...InputOption) *Input {
	i := &Input{
		shape:       tf.MakeShape(0),
		batchSize:   -1,
		dtype:       Float32,
		inputTensor: nil,
		sparse:      false,
		ragged:      false,
		name:        uniqueName("input"),
	}
	for _, option := range options {
		option(i)
	}
	return i
}

type InputOption func(*Input)

func InputWithName(name string) func(i *Input) {
	return func(i *Input) {
		i.name = name
	}
}

func InputWithTrainable(trainable bool) func(i *Input) {
	return func(i *Input) {
		i.trainable = trainable
	}
}

func InputWithInputShape(inputShape tf.Shape) func(i *Input) {
	return func(i *Input) {
		i.shape = inputShape
	}
}

func InputWithBatchSize(batchSize float64) func(i *Input) {
	return func(i *Input) {
		i.batchSize = batchSize
	}
}

func InputWithDtype(dtype DataType) func(i *Input) {
	return func(i *Input) {
		i.dtype = dtype
	}
}

func InputWithInputTensor(inputTensor interface{}) func(i *Input) {
	return func(i *Input) {
		i.inputTensor = inputTensor
	}
}

func InputWithSparse(sparse bool) func(i *Input) {
	return func(i *Input) {
		i.sparse = sparse
	}
}

func InputWithRagged(ragged bool) func(i *Input) {
	return func(i *Input) {
		i.ragged = ragged
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

type jsonConfigInput struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes []interface{}          `json:"inbound_nodes"`
}

func (i *Input) GetKerasLayerConfig() interface{} {
	var shape []interface{}

	dims, _ := i.shape.ToSlice()

	for _, dim := range dims {
		if dim == -1 {
			shape = append(shape, nil)
		} else {
			shape = append(shape, dim)
		}
	}

	return jsonConfigInput{
		ClassName: "InputLayer",
		Name:      i.name,
		Config: map[string]interface{}{
			"name":              i.name,
			"batch_input_shape": shape,
			"dtype":             i.dtype.String(),
			"sparse":            i.sparse,
			"ragged":            i.ragged,
		},
		InboundNodes: []interface{}{},
	}
}

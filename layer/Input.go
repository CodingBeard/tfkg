package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LInput struct {
	name        string
	dtype       DataType
	shape       tf.Shape
	trainable   bool
	batchSize   float64
	inputTensor interface{}
	sparse      bool
	ragged      bool
}

func Input() *LInput {
	i := &LInput{
		batchSize:   -1,
		dtype:       Float32,
		inputTensor: nil,
		sparse:      false,
		ragged:      false,
		name:        UniqueName("input"),
	}
	return i
}

func (i *LInput) SetName(name string) *LInput {
	i.name = name
	return i
}

func (i *LInput) SetTrainable(trainable bool) *LInput {
	i.trainable = trainable
	return i
}

func (i *LInput) SetInputShape(inputShape tf.Shape) *LInput {
	i.shape = inputShape
	return i
}

func (i *LInput) SetBatchSize(batchSize float64) *LInput {
	i.batchSize = batchSize
	return i
}

func (i *LInput) SetDtype(dtype DataType) *LInput {
	i.dtype = dtype
	return i
}
func (i *LInput) SetSparse(sparse bool) *LInput {
	i.sparse = sparse
	return i
}

func (i *LInput) SetRagged(ragged bool) *LInput {
	i.ragged = ragged
	return i
}

func (i *LInput) GetShape() tf.Shape {
	return i.shape
}

func (i *LInput) GetDtype() DataType {
	return i.dtype
}

func (i *LInput) SetInputs(inputs ...Layer) Layer {
	return i
}

func (i *LInput) GetInputs() []Layer {
	return []Layer{}
}

func (i *LInput) GetName() string {
	return i.name
}

type jsonConfigInput struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes []interface{}          `json:"inbound_nodes"`
}

func (i *LInput) GetKerasLayerConfig() interface{} {
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

func (i *LInput) GetCustomLayerDefinition() string {
	return ``
}

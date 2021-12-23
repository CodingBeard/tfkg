package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type RandomFourierFeatures struct {
	name              string
	dtype             DataType
	inputs            []Layer
	shape             tf.Shape
	trainable         bool
	outputDim         float64
	kernelInitializer string
	scale             interface{}
}

func NewRandomFourierFeatures(outputDim float64, options ...RandomFourierFeaturesOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		r := &RandomFourierFeatures{
			outputDim:         outputDim,
			kernelInitializer: "gaussian",
			scale:             nil,
			trainable:         true,
			inputs:            inputs,
			name:              UniqueName("randomfourierfeatures"),
		}
		for _, option := range options {
			option(r)
		}
		return r
	}
}

type RandomFourierFeaturesOption func(*RandomFourierFeatures)

func RandomFourierFeaturesWithName(name string) func(r *RandomFourierFeatures) {
	return func(r *RandomFourierFeatures) {
		r.name = name
	}
}

func RandomFourierFeaturesWithDtype(dtype DataType) func(r *RandomFourierFeatures) {
	return func(r *RandomFourierFeatures) {
		r.dtype = dtype
	}
}

func RandomFourierFeaturesWithTrainable(trainable bool) func(r *RandomFourierFeatures) {
	return func(r *RandomFourierFeatures) {
		r.trainable = trainable
	}
}

func RandomFourierFeaturesWithKernelInitializer(kernelInitializer string) func(r *RandomFourierFeatures) {
	return func(r *RandomFourierFeatures) {
		r.kernelInitializer = kernelInitializer
	}
}

func RandomFourierFeaturesWithScale(scale interface{}) func(r *RandomFourierFeatures) {
	return func(r *RandomFourierFeatures) {
		r.scale = scale
	}
}

func (r *RandomFourierFeatures) GetShape() tf.Shape {
	return r.shape
}

func (r *RandomFourierFeatures) GetDtype() DataType {
	return r.dtype
}

func (r *RandomFourierFeatures) SetInput(inputs []Layer) {
	r.inputs = inputs
	r.dtype = inputs[0].GetDtype()
}

func (r *RandomFourierFeatures) GetInputs() []Layer {
	return r.inputs
}

func (r *RandomFourierFeatures) GetName() string {
	return r.name
}

type jsonConfigRandomFourierFeatures struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (r *RandomFourierFeatures) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range r.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigRandomFourierFeatures{
		ClassName: "RandomFourierFeatures",
		Name:      r.name,
		Config: map[string]interface{}{
			"dtype":              r.dtype.String(),
			"kernel_initializer": r.kernelInitializer,
			"name":               r.name,
			"output_dim":         r.outputDim,
			"scale":              r.scale,
			"trainable":          r.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (r *RandomFourierFeatures) GetCustomLayerDefinition() string {
	return ``
}

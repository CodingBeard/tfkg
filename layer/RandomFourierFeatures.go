package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LRandomFourierFeatures struct {
	dtype             DataType
	inputs            []Layer
	kernelInitializer string
	name              string
	outputDim         float64
	scale             interface{}
	shape             tf.Shape
	trainable         bool
	layerWeights      []*tf.Tensor
}

func RandomFourierFeatures(outputDim float64) *LRandomFourierFeatures {
	return &LRandomFourierFeatures{
		dtype:             Float32,
		kernelInitializer: "gaussian",
		name:              UniqueName("nil"),
		outputDim:         outputDim,
		scale:             nil,
		trainable:         true,
	}
}

func (l *LRandomFourierFeatures) SetDtype(dtype DataType) *LRandomFourierFeatures {
	l.dtype = dtype
	return l
}

func (l *LRandomFourierFeatures) SetKernelInitializer(kernelInitializer string) *LRandomFourierFeatures {
	l.kernelInitializer = kernelInitializer
	return l
}

func (l *LRandomFourierFeatures) SetName(name string) *LRandomFourierFeatures {
	l.name = name
	return l
}

func (l *LRandomFourierFeatures) SetScale(scale interface{}) *LRandomFourierFeatures {
	l.scale = scale
	return l
}

func (l *LRandomFourierFeatures) SetShape(shape tf.Shape) *LRandomFourierFeatures {
	l.shape = shape
	return l
}

func (l *LRandomFourierFeatures) SetTrainable(trainable bool) *LRandomFourierFeatures {
	l.trainable = trainable
	return l
}

func (l *LRandomFourierFeatures) SetLayerWeights(layerWeights []*tf.Tensor) *LRandomFourierFeatures {
	l.layerWeights = layerWeights
	return l
}

func (l *LRandomFourierFeatures) GetShape() tf.Shape {
	return l.shape
}

func (l *LRandomFourierFeatures) GetDtype() DataType {
	return l.dtype
}

func (l *LRandomFourierFeatures) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LRandomFourierFeatures) GetInputs() []Layer {
	return l.inputs
}

func (l *LRandomFourierFeatures) GetName() string {
	return l.name
}

func (l *LRandomFourierFeatures) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLRandomFourierFeatures struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LRandomFourierFeatures) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLRandomFourierFeatures{
		ClassName: "RandomFourierFeatures",
		Name:      l.name,
		Config: map[string]interface{}{
			"dtype":              l.dtype.String(),
			"kernel_initializer": l.kernelInitializer,
			"name":               l.name,
			"output_dim":         l.outputDim,
			"scale":              l.scale,
			"trainable":          l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LRandomFourierFeatures) GetCustomLayerDefinition() string {
	return ``
}

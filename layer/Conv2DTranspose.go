package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LConv2DTranspose struct {
	activation          string
	activityRegularizer regularizer.Regularizer
	biasConstraint      constraint.Constraint
	biasInitializer     initializer.Initializer
	biasRegularizer     regularizer.Regularizer
	dataFormat          interface{}
	dilationRate        []interface{}
	dtype               DataType
	filters             float64
	groups              float64
	inputs              []Layer
	kernelConstraint    constraint.Constraint
	kernelInitializer   initializer.Initializer
	kernelRegularizer   regularizer.Regularizer
	kernelSize          float64
	name                string
	outputPadding       interface{}
	padding             string
	shape               tf.Shape
	strides             []interface{}
	trainable           bool
	useBias             bool
	layerWeights        interface{}
}

func Conv2DTranspose(filters float64, kernelSize float64) *LConv2DTranspose {
	return &LConv2DTranspose{
		activation:          "linear",
		activityRegularizer: &regularizer.NilRegularizer{},
		biasConstraint:      &constraint.NilConstraint{},
		biasInitializer:     initializer.Zeros(),
		biasRegularizer:     &regularizer.NilRegularizer{},
		dataFormat:          nil,
		dilationRate:        []interface{}{1, 1},
		dtype:               Float32,
		filters:             filters,
		groups:              1,
		kernelConstraint:    &constraint.NilConstraint{},
		kernelInitializer:   initializer.GlorotUniform(),
		kernelRegularizer:   &regularizer.NilRegularizer{},
		kernelSize:          kernelSize,
		name:                UniqueName("conv2d_transpose"),
		outputPadding:       nil,
		padding:             "valid",
		strides:             []interface{}{1, 1},
		trainable:           true,
		useBias:             true,
	}
}

func (l *LConv2DTranspose) SetActivation(activation string) *LConv2DTranspose {
	l.activation = activation
	return l
}

func (l *LConv2DTranspose) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LConv2DTranspose {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LConv2DTranspose) SetBiasConstraint(biasConstraint constraint.Constraint) *LConv2DTranspose {
	l.biasConstraint = biasConstraint
	return l
}

func (l *LConv2DTranspose) SetBiasInitializer(biasInitializer initializer.Initializer) *LConv2DTranspose {
	l.biasInitializer = biasInitializer
	return l
}

func (l *LConv2DTranspose) SetBiasRegularizer(biasRegularizer regularizer.Regularizer) *LConv2DTranspose {
	l.biasRegularizer = biasRegularizer
	return l
}

func (l *LConv2DTranspose) SetDataFormat(dataFormat interface{}) *LConv2DTranspose {
	l.dataFormat = dataFormat
	return l
}

func (l *LConv2DTranspose) SetDilationRate(dilationRate []interface{}) *LConv2DTranspose {
	l.dilationRate = dilationRate
	return l
}

func (l *LConv2DTranspose) SetDtype(dtype DataType) *LConv2DTranspose {
	l.dtype = dtype
	return l
}

func (l *LConv2DTranspose) SetGroups(groups float64) *LConv2DTranspose {
	l.groups = groups
	return l
}

func (l *LConv2DTranspose) SetKernelConstraint(kernelConstraint constraint.Constraint) *LConv2DTranspose {
	l.kernelConstraint = kernelConstraint
	return l
}

func (l *LConv2DTranspose) SetKernelInitializer(kernelInitializer initializer.Initializer) *LConv2DTranspose {
	l.kernelInitializer = kernelInitializer
	return l
}

func (l *LConv2DTranspose) SetKernelRegularizer(kernelRegularizer regularizer.Regularizer) *LConv2DTranspose {
	l.kernelRegularizer = kernelRegularizer
	return l
}

func (l *LConv2DTranspose) SetName(name string) *LConv2DTranspose {
	l.name = name
	return l
}

func (l *LConv2DTranspose) SetOutputPadding(outputPadding interface{}) *LConv2DTranspose {
	l.outputPadding = outputPadding
	return l
}

func (l *LConv2DTranspose) SetPadding(padding string) *LConv2DTranspose {
	l.padding = padding
	return l
}

func (l *LConv2DTranspose) SetShape(shape tf.Shape) *LConv2DTranspose {
	l.shape = shape
	return l
}

func (l *LConv2DTranspose) SetStrides(strides []interface{}) *LConv2DTranspose {
	l.strides = strides
	return l
}

func (l *LConv2DTranspose) SetTrainable(trainable bool) *LConv2DTranspose {
	l.trainable = trainable
	return l
}

func (l *LConv2DTranspose) SetUseBias(useBias bool) *LConv2DTranspose {
	l.useBias = useBias
	return l
}

func (l *LConv2DTranspose) SetLayerWeights(layerWeights interface{}) *LConv2DTranspose {
	l.layerWeights = layerWeights
	return l
}

func (l *LConv2DTranspose) GetShape() tf.Shape {
	return l.shape
}

func (l *LConv2DTranspose) GetDtype() DataType {
	return l.dtype
}

func (l *LConv2DTranspose) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LConv2DTranspose) GetInputs() []Layer {
	return l.inputs
}

func (l *LConv2DTranspose) GetName() string {
	return l.name
}

func (l *LConv2DTranspose) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLConv2DTranspose struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LConv2DTranspose) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLConv2DTranspose{
		ClassName: "Conv2DTranspose",
		Name:      l.name,
		Config: map[string]interface{}{
			"activation":           l.activation,
			"activity_regularizer": l.activityRegularizer.GetKerasLayerConfig(),
			"bias_constraint":      l.biasConstraint.GetKerasLayerConfig(),
			"bias_initializer":     l.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer":     l.biasRegularizer.GetKerasLayerConfig(),
			"data_format":          l.dataFormat,
			"dilation_rate":        l.dilationRate,
			"dtype":                l.dtype.String(),
			"filters":              l.filters,
			"groups":               l.groups,
			"kernel_constraint":    l.kernelConstraint.GetKerasLayerConfig(),
			"kernel_initializer":   l.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer":   l.kernelRegularizer.GetKerasLayerConfig(),
			"kernel_size":          l.kernelSize,
			"name":                 l.name,
			"output_padding":       l.outputPadding,
			"padding":              l.padding,
			"strides":              l.strides,
			"trainable":            l.trainable,
			"use_bias":             l.useBias,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LConv2DTranspose) GetCustomLayerDefinition() string {
	return ``
}

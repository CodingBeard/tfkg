package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LConv3DTranspose struct {
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

func Conv3DTranspose(filters float64, kernelSize float64) *LConv3DTranspose {
	return &LConv3DTranspose{
		activation:          "linear",
		activityRegularizer: &regularizer.NilRegularizer{},
		biasConstraint:      &constraint.NilConstraint{},
		biasInitializer:     initializer.Zeros(),
		biasRegularizer:     &regularizer.NilRegularizer{},
		dataFormat:          nil,
		dilationRate:        []interface{}{1, 1, 1},
		dtype:               Float32,
		filters:             filters,
		groups:              1,
		kernelConstraint:    &constraint.NilConstraint{},
		kernelInitializer:   initializer.GlorotUniform(),
		kernelRegularizer:   &regularizer.NilRegularizer{},
		kernelSize:          kernelSize,
		name:                UniqueName("conv3d_transpose"),
		outputPadding:       nil,
		padding:             "valid",
		strides:             []interface{}{1, 1, 1},
		trainable:           true,
		useBias:             true,
	}
}

func (l *LConv3DTranspose) SetActivation(activation string) *LConv3DTranspose {
	l.activation = activation
	return l
}

func (l *LConv3DTranspose) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LConv3DTranspose {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LConv3DTranspose) SetBiasConstraint(biasConstraint constraint.Constraint) *LConv3DTranspose {
	l.biasConstraint = biasConstraint
	return l
}

func (l *LConv3DTranspose) SetBiasInitializer(biasInitializer initializer.Initializer) *LConv3DTranspose {
	l.biasInitializer = biasInitializer
	return l
}

func (l *LConv3DTranspose) SetBiasRegularizer(biasRegularizer regularizer.Regularizer) *LConv3DTranspose {
	l.biasRegularizer = biasRegularizer
	return l
}

func (l *LConv3DTranspose) SetDataFormat(dataFormat interface{}) *LConv3DTranspose {
	l.dataFormat = dataFormat
	return l
}

func (l *LConv3DTranspose) SetDilationRate(dilationRate []interface{}) *LConv3DTranspose {
	l.dilationRate = dilationRate
	return l
}

func (l *LConv3DTranspose) SetDtype(dtype DataType) *LConv3DTranspose {
	l.dtype = dtype
	return l
}

func (l *LConv3DTranspose) SetGroups(groups float64) *LConv3DTranspose {
	l.groups = groups
	return l
}

func (l *LConv3DTranspose) SetKernelConstraint(kernelConstraint constraint.Constraint) *LConv3DTranspose {
	l.kernelConstraint = kernelConstraint
	return l
}

func (l *LConv3DTranspose) SetKernelInitializer(kernelInitializer initializer.Initializer) *LConv3DTranspose {
	l.kernelInitializer = kernelInitializer
	return l
}

func (l *LConv3DTranspose) SetKernelRegularizer(kernelRegularizer regularizer.Regularizer) *LConv3DTranspose {
	l.kernelRegularizer = kernelRegularizer
	return l
}

func (l *LConv3DTranspose) SetName(name string) *LConv3DTranspose {
	l.name = name
	return l
}

func (l *LConv3DTranspose) SetOutputPadding(outputPadding interface{}) *LConv3DTranspose {
	l.outputPadding = outputPadding
	return l
}

func (l *LConv3DTranspose) SetPadding(padding string) *LConv3DTranspose {
	l.padding = padding
	return l
}

func (l *LConv3DTranspose) SetShape(shape tf.Shape) *LConv3DTranspose {
	l.shape = shape
	return l
}

func (l *LConv3DTranspose) SetStrides(strides []interface{}) *LConv3DTranspose {
	l.strides = strides
	return l
}

func (l *LConv3DTranspose) SetTrainable(trainable bool) *LConv3DTranspose {
	l.trainable = trainable
	return l
}

func (l *LConv3DTranspose) SetUseBias(useBias bool) *LConv3DTranspose {
	l.useBias = useBias
	return l
}

func (l *LConv3DTranspose) SetLayerWeights(layerWeights interface{}) *LConv3DTranspose {
	l.layerWeights = layerWeights
	return l
}

func (l *LConv3DTranspose) GetShape() tf.Shape {
	return l.shape
}

func (l *LConv3DTranspose) GetDtype() DataType {
	return l.dtype
}

func (l *LConv3DTranspose) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LConv3DTranspose) GetInputs() []Layer {
	return l.inputs
}

func (l *LConv3DTranspose) GetName() string {
	return l.name
}

func (l *LConv3DTranspose) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLConv3DTranspose struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LConv3DTranspose) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLConv3DTranspose{
		ClassName: "Conv3DTranspose",
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

func (l *LConv3DTranspose) GetCustomLayerDefinition() string {
	return ``
}

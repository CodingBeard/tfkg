package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LConv2D struct {
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
	padding             string
	shape               tf.Shape
	strides             []interface{}
	trainable           bool
	useBias             bool
	layerWeights        interface{}
}

func Conv2D(filters float64, kernelSize float64) *LConv2D {
	return &LConv2D{
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
		name:                UniqueName("conv2d"),
		padding:             "valid",
		strides:             []interface{}{1, 1},
		trainable:           true,
		useBias:             true,
	}
}

func (l *LConv2D) SetActivation(activation string) *LConv2D {
	l.activation = activation
	return l
}

func (l *LConv2D) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LConv2D {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LConv2D) SetBiasConstraint(biasConstraint constraint.Constraint) *LConv2D {
	l.biasConstraint = biasConstraint
	return l
}

func (l *LConv2D) SetBiasInitializer(biasInitializer initializer.Initializer) *LConv2D {
	l.biasInitializer = biasInitializer
	return l
}

func (l *LConv2D) SetBiasRegularizer(biasRegularizer regularizer.Regularizer) *LConv2D {
	l.biasRegularizer = biasRegularizer
	return l
}

func (l *LConv2D) SetDataFormat(dataFormat interface{}) *LConv2D {
	l.dataFormat = dataFormat
	return l
}

func (l *LConv2D) SetDilationRate(dilationRate []interface{}) *LConv2D {
	l.dilationRate = dilationRate
	return l
}

func (l *LConv2D) SetDtype(dtype DataType) *LConv2D {
	l.dtype = dtype
	return l
}

func (l *LConv2D) SetGroups(groups float64) *LConv2D {
	l.groups = groups
	return l
}

func (l *LConv2D) SetKernelConstraint(kernelConstraint constraint.Constraint) *LConv2D {
	l.kernelConstraint = kernelConstraint
	return l
}

func (l *LConv2D) SetKernelInitializer(kernelInitializer initializer.Initializer) *LConv2D {
	l.kernelInitializer = kernelInitializer
	return l
}

func (l *LConv2D) SetKernelRegularizer(kernelRegularizer regularizer.Regularizer) *LConv2D {
	l.kernelRegularizer = kernelRegularizer
	return l
}

func (l *LConv2D) SetName(name string) *LConv2D {
	l.name = name
	return l
}

func (l *LConv2D) SetPadding(padding string) *LConv2D {
	l.padding = padding
	return l
}

func (l *LConv2D) SetShape(shape tf.Shape) *LConv2D {
	l.shape = shape
	return l
}

func (l *LConv2D) SetStrides(strides []interface{}) *LConv2D {
	l.strides = strides
	return l
}

func (l *LConv2D) SetTrainable(trainable bool) *LConv2D {
	l.trainable = trainable
	return l
}

func (l *LConv2D) SetUseBias(useBias bool) *LConv2D {
	l.useBias = useBias
	return l
}

func (l *LConv2D) SetLayerWeights(layerWeights interface{}) *LConv2D {
	l.layerWeights = layerWeights
	return l
}

func (l *LConv2D) GetShape() tf.Shape {
	return l.shape
}

func (l *LConv2D) GetDtype() DataType {
	return l.dtype
}

func (l *LConv2D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LConv2D) GetInputs() []Layer {
	return l.inputs
}

func (l *LConv2D) GetName() string {
	return l.name
}

func (l *LConv2D) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLConv2D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LConv2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLConv2D{
		ClassName: "Conv2D",
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
			"padding":              l.padding,
			"strides":              l.strides,
			"trainable":            l.trainable,
			"use_bias":             l.useBias,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LConv2D) GetCustomLayerDefinition() string {
	return ``
}

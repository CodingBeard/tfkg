package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LSeparableConv2D struct {
	activation           string
	activityRegularizer  regularizer.Regularizer
	biasConstraint       constraint.Constraint
	biasInitializer      initializer.Initializer
	biasRegularizer      regularizer.Regularizer
	dataFormat           interface{}
	depthMultiplier      float64
	depthwiseConstraint  constraint.Constraint
	depthwiseInitializer initializer.Initializer
	depthwiseRegularizer regularizer.Regularizer
	dilationRate         []interface{}
	dtype                DataType
	filters              float64
	groups               float64
	inputs               []Layer
	kernelConstraint     constraint.Constraint
	kernelInitializer    initializer.Initializer
	kernelRegularizer    regularizer.Regularizer
	kernelSize           float64
	name                 string
	padding              string
	pointwiseConstraint  constraint.Constraint
	pointwiseInitializer initializer.Initializer
	pointwiseRegularizer regularizer.Regularizer
	shape                tf.Shape
	strides              []interface{}
	trainable            bool
	useBias              bool
}

func SeparableConv2D(filters float64, kernelSize float64) *LSeparableConv2D {
	return &LSeparableConv2D{
		activation:           "linear",
		activityRegularizer:  &regularizer.NilRegularizer{},
		biasConstraint:       &constraint.NilConstraint{},
		biasInitializer:      initializer.Zeros(),
		biasRegularizer:      &regularizer.NilRegularizer{},
		dataFormat:           nil,
		depthMultiplier:      1,
		depthwiseConstraint:  &constraint.NilConstraint{},
		depthwiseInitializer: initializer.GlorotUniform(),
		depthwiseRegularizer: &regularizer.NilRegularizer{},
		dilationRate:         []interface{}{1, 1},
		dtype:                Float32,
		filters:              filters,
		groups:               1,
		kernelConstraint:     &constraint.NilConstraint{},
		kernelInitializer:    initializer.GlorotUniform(),
		kernelRegularizer:    &regularizer.NilRegularizer{},
		kernelSize:           kernelSize,
		name:                 UniqueName("separable_conv2d"),
		padding:              "valid",
		pointwiseConstraint:  &constraint.NilConstraint{},
		pointwiseInitializer: initializer.GlorotUniform(),
		pointwiseRegularizer: &regularizer.NilRegularizer{},
		strides:              []interface{}{1, 1},
		trainable:            true,
		useBias:              true,
	}
}

func (l *LSeparableConv2D) SetActivation(activation string) *LSeparableConv2D {
	l.activation = activation
	return l
}

func (l *LSeparableConv2D) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LSeparableConv2D {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LSeparableConv2D) SetBiasConstraint(biasConstraint constraint.Constraint) *LSeparableConv2D {
	l.biasConstraint = biasConstraint
	return l
}

func (l *LSeparableConv2D) SetBiasInitializer(biasInitializer initializer.Initializer) *LSeparableConv2D {
	l.biasInitializer = biasInitializer
	return l
}

func (l *LSeparableConv2D) SetBiasRegularizer(biasRegularizer regularizer.Regularizer) *LSeparableConv2D {
	l.biasRegularizer = biasRegularizer
	return l
}

func (l *LSeparableConv2D) SetDataFormat(dataFormat interface{}) *LSeparableConv2D {
	l.dataFormat = dataFormat
	return l
}

func (l *LSeparableConv2D) SetDepthMultiplier(depthMultiplier float64) *LSeparableConv2D {
	l.depthMultiplier = depthMultiplier
	return l
}

func (l *LSeparableConv2D) SetDepthwiseConstraint(depthwiseConstraint constraint.Constraint) *LSeparableConv2D {
	l.depthwiseConstraint = depthwiseConstraint
	return l
}

func (l *LSeparableConv2D) SetDepthwiseInitializer(depthwiseInitializer initializer.Initializer) *LSeparableConv2D {
	l.depthwiseInitializer = depthwiseInitializer
	return l
}

func (l *LSeparableConv2D) SetDepthwiseRegularizer(depthwiseRegularizer regularizer.Regularizer) *LSeparableConv2D {
	l.depthwiseRegularizer = depthwiseRegularizer
	return l
}

func (l *LSeparableConv2D) SetDilationRate(dilationRate []interface{}) *LSeparableConv2D {
	l.dilationRate = dilationRate
	return l
}

func (l *LSeparableConv2D) SetDtype(dtype DataType) *LSeparableConv2D {
	l.dtype = dtype
	return l
}

func (l *LSeparableConv2D) SetGroups(groups float64) *LSeparableConv2D {
	l.groups = groups
	return l
}

func (l *LSeparableConv2D) SetKernelConstraint(kernelConstraint constraint.Constraint) *LSeparableConv2D {
	l.kernelConstraint = kernelConstraint
	return l
}

func (l *LSeparableConv2D) SetKernelInitializer(kernelInitializer initializer.Initializer) *LSeparableConv2D {
	l.kernelInitializer = kernelInitializer
	return l
}

func (l *LSeparableConv2D) SetKernelRegularizer(kernelRegularizer regularizer.Regularizer) *LSeparableConv2D {
	l.kernelRegularizer = kernelRegularizer
	return l
}

func (l *LSeparableConv2D) SetName(name string) *LSeparableConv2D {
	l.name = name
	return l
}

func (l *LSeparableConv2D) SetPadding(padding string) *LSeparableConv2D {
	l.padding = padding
	return l
}

func (l *LSeparableConv2D) SetPointwiseConstraint(pointwiseConstraint constraint.Constraint) *LSeparableConv2D {
	l.pointwiseConstraint = pointwiseConstraint
	return l
}

func (l *LSeparableConv2D) SetPointwiseInitializer(pointwiseInitializer initializer.Initializer) *LSeparableConv2D {
	l.pointwiseInitializer = pointwiseInitializer
	return l
}

func (l *LSeparableConv2D) SetPointwiseRegularizer(pointwiseRegularizer regularizer.Regularizer) *LSeparableConv2D {
	l.pointwiseRegularizer = pointwiseRegularizer
	return l
}

func (l *LSeparableConv2D) SetShape(shape tf.Shape) *LSeparableConv2D {
	l.shape = shape
	return l
}

func (l *LSeparableConv2D) SetStrides(strides []interface{}) *LSeparableConv2D {
	l.strides = strides
	return l
}

func (l *LSeparableConv2D) SetTrainable(trainable bool) *LSeparableConv2D {
	l.trainable = trainable
	return l
}

func (l *LSeparableConv2D) SetUseBias(useBias bool) *LSeparableConv2D {
	l.useBias = useBias
	return l
}

func (l *LSeparableConv2D) GetShape() tf.Shape {
	return l.shape
}

func (l *LSeparableConv2D) GetDtype() DataType {
	return l.dtype
}

func (l *LSeparableConv2D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LSeparableConv2D) GetInputs() []Layer {
	return l.inputs
}

func (l *LSeparableConv2D) GetName() string {
	return l.name
}

type jsonConfigLSeparableConv2D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LSeparableConv2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLSeparableConv2D{
		ClassName: "SeparableConv2D",
		Name:      l.name,
		Config: map[string]interface{}{
			"activation":            l.activation,
			"activity_regularizer":  l.activityRegularizer.GetKerasLayerConfig(),
			"bias_constraint":       l.biasConstraint.GetKerasLayerConfig(),
			"bias_initializer":      l.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer":      l.biasRegularizer.GetKerasLayerConfig(),
			"data_format":           l.dataFormat,
			"depth_multiplier":      l.depthMultiplier,
			"depthwise_constraint":  l.depthwiseConstraint.GetKerasLayerConfig(),
			"depthwise_initializer": l.depthwiseInitializer.GetKerasLayerConfig(),
			"depthwise_regularizer": l.depthwiseRegularizer.GetKerasLayerConfig(),
			"dilation_rate":         l.dilationRate,
			"dtype":                 l.dtype.String(),
			"filters":               l.filters,
			"groups":                l.groups,
			"kernel_constraint":     l.kernelConstraint.GetKerasLayerConfig(),
			"kernel_initializer":    l.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer":    l.kernelRegularizer.GetKerasLayerConfig(),
			"kernel_size":           l.kernelSize,
			"name":                  l.name,
			"padding":               l.padding,
			"pointwise_constraint":  l.pointwiseConstraint.GetKerasLayerConfig(),
			"pointwise_initializer": l.pointwiseInitializer.GetKerasLayerConfig(),
			"pointwise_regularizer": l.pointwiseRegularizer.GetKerasLayerConfig(),
			"strides":               l.strides,
			"trainable":             l.trainable,
			"use_bias":              l.useBias,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LSeparableConv2D) GetCustomLayerDefinition() string {
	return ``
}

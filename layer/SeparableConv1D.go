package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LSeparableConv1D struct {
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
	dilationRate         float64
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
	strides              float64
	trainable            bool
	useBias              bool
}

func SeparableConv1D(filters float64, kernelSize float64) *LSeparableConv1D {
	return &LSeparableConv1D{
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
		dilationRate:         1,
		dtype:                Float32,
		filters:              filters,
		groups:               1,
		kernelConstraint:     &constraint.NilConstraint{},
		kernelInitializer:    initializer.GlorotUniform(),
		kernelRegularizer:    &regularizer.NilRegularizer{},
		kernelSize:           kernelSize,
		name:                 UniqueName("separable_conv1d"),
		padding:              "valid",
		pointwiseConstraint:  &constraint.NilConstraint{},
		pointwiseInitializer: initializer.GlorotUniform(),
		pointwiseRegularizer: &regularizer.NilRegularizer{},
		strides:              1,
		trainable:            true,
		useBias:              true,
	}
}

func (l *LSeparableConv1D) SetActivation(activation string) *LSeparableConv1D {
	l.activation = activation
	return l
}

func (l *LSeparableConv1D) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LSeparableConv1D {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LSeparableConv1D) SetBiasConstraint(biasConstraint constraint.Constraint) *LSeparableConv1D {
	l.biasConstraint = biasConstraint
	return l
}

func (l *LSeparableConv1D) SetBiasInitializer(biasInitializer initializer.Initializer) *LSeparableConv1D {
	l.biasInitializer = biasInitializer
	return l
}

func (l *LSeparableConv1D) SetBiasRegularizer(biasRegularizer regularizer.Regularizer) *LSeparableConv1D {
	l.biasRegularizer = biasRegularizer
	return l
}

func (l *LSeparableConv1D) SetDataFormat(dataFormat interface{}) *LSeparableConv1D {
	l.dataFormat = dataFormat
	return l
}

func (l *LSeparableConv1D) SetDepthMultiplier(depthMultiplier float64) *LSeparableConv1D {
	l.depthMultiplier = depthMultiplier
	return l
}

func (l *LSeparableConv1D) SetDepthwiseConstraint(depthwiseConstraint constraint.Constraint) *LSeparableConv1D {
	l.depthwiseConstraint = depthwiseConstraint
	return l
}

func (l *LSeparableConv1D) SetDepthwiseInitializer(depthwiseInitializer initializer.Initializer) *LSeparableConv1D {
	l.depthwiseInitializer = depthwiseInitializer
	return l
}

func (l *LSeparableConv1D) SetDepthwiseRegularizer(depthwiseRegularizer regularizer.Regularizer) *LSeparableConv1D {
	l.depthwiseRegularizer = depthwiseRegularizer
	return l
}

func (l *LSeparableConv1D) SetDilationRate(dilationRate float64) *LSeparableConv1D {
	l.dilationRate = dilationRate
	return l
}

func (l *LSeparableConv1D) SetDtype(dtype DataType) *LSeparableConv1D {
	l.dtype = dtype
	return l
}

func (l *LSeparableConv1D) SetGroups(groups float64) *LSeparableConv1D {
	l.groups = groups
	return l
}

func (l *LSeparableConv1D) SetKernelConstraint(kernelConstraint constraint.Constraint) *LSeparableConv1D {
	l.kernelConstraint = kernelConstraint
	return l
}

func (l *LSeparableConv1D) SetKernelInitializer(kernelInitializer initializer.Initializer) *LSeparableConv1D {
	l.kernelInitializer = kernelInitializer
	return l
}

func (l *LSeparableConv1D) SetKernelRegularizer(kernelRegularizer regularizer.Regularizer) *LSeparableConv1D {
	l.kernelRegularizer = kernelRegularizer
	return l
}

func (l *LSeparableConv1D) SetName(name string) *LSeparableConv1D {
	l.name = name
	return l
}

func (l *LSeparableConv1D) SetPadding(padding string) *LSeparableConv1D {
	l.padding = padding
	return l
}

func (l *LSeparableConv1D) SetPointwiseConstraint(pointwiseConstraint constraint.Constraint) *LSeparableConv1D {
	l.pointwiseConstraint = pointwiseConstraint
	return l
}

func (l *LSeparableConv1D) SetPointwiseInitializer(pointwiseInitializer initializer.Initializer) *LSeparableConv1D {
	l.pointwiseInitializer = pointwiseInitializer
	return l
}

func (l *LSeparableConv1D) SetPointwiseRegularizer(pointwiseRegularizer regularizer.Regularizer) *LSeparableConv1D {
	l.pointwiseRegularizer = pointwiseRegularizer
	return l
}

func (l *LSeparableConv1D) SetShape(shape tf.Shape) *LSeparableConv1D {
	l.shape = shape
	return l
}

func (l *LSeparableConv1D) SetStrides(strides float64) *LSeparableConv1D {
	l.strides = strides
	return l
}

func (l *LSeparableConv1D) SetTrainable(trainable bool) *LSeparableConv1D {
	l.trainable = trainable
	return l
}

func (l *LSeparableConv1D) SetUseBias(useBias bool) *LSeparableConv1D {
	l.useBias = useBias
	return l
}

func (l *LSeparableConv1D) GetShape() tf.Shape {
	return l.shape
}

func (l *LSeparableConv1D) GetDtype() DataType {
	return l.dtype
}

func (l *LSeparableConv1D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LSeparableConv1D) GetInputs() []Layer {
	return l.inputs
}

func (l *LSeparableConv1D) GetName() string {
	return l.name
}

type jsonConfigLSeparableConv1D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LSeparableConv1D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLSeparableConv1D{
		ClassName: "SeparableConv1D",
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

func (l *LSeparableConv1D) GetCustomLayerDefinition() string {
	return ``
}

package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LDepthwiseConv2D struct {
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
	groups               float64
	inputs               []Layer
	kernelSize           float64
	name                 string
	padding              string
	shape                tf.Shape
	strides              []interface{}
	trainable            bool
	useBias              bool
	layerWeights         interface{}
}

func DepthwiseConv2D(kernelSize float64) *LDepthwiseConv2D {
	return &LDepthwiseConv2D{
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
		groups:               1,
		kernelSize:           kernelSize,
		name:                 UniqueName("depthwise_conv2d"),
		padding:              "valid",
		strides:              []interface{}{1, 1},
		trainable:            true,
		useBias:              true,
	}
}

func (l *LDepthwiseConv2D) SetActivation(activation string) *LDepthwiseConv2D {
	l.activation = activation
	return l
}

func (l *LDepthwiseConv2D) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LDepthwiseConv2D {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LDepthwiseConv2D) SetBiasConstraint(biasConstraint constraint.Constraint) *LDepthwiseConv2D {
	l.biasConstraint = biasConstraint
	return l
}

func (l *LDepthwiseConv2D) SetBiasInitializer(biasInitializer initializer.Initializer) *LDepthwiseConv2D {
	l.biasInitializer = biasInitializer
	return l
}

func (l *LDepthwiseConv2D) SetBiasRegularizer(biasRegularizer regularizer.Regularizer) *LDepthwiseConv2D {
	l.biasRegularizer = biasRegularizer
	return l
}

func (l *LDepthwiseConv2D) SetDataFormat(dataFormat interface{}) *LDepthwiseConv2D {
	l.dataFormat = dataFormat
	return l
}

func (l *LDepthwiseConv2D) SetDepthMultiplier(depthMultiplier float64) *LDepthwiseConv2D {
	l.depthMultiplier = depthMultiplier
	return l
}

func (l *LDepthwiseConv2D) SetDepthwiseConstraint(depthwiseConstraint constraint.Constraint) *LDepthwiseConv2D {
	l.depthwiseConstraint = depthwiseConstraint
	return l
}

func (l *LDepthwiseConv2D) SetDepthwiseInitializer(depthwiseInitializer initializer.Initializer) *LDepthwiseConv2D {
	l.depthwiseInitializer = depthwiseInitializer
	return l
}

func (l *LDepthwiseConv2D) SetDepthwiseRegularizer(depthwiseRegularizer regularizer.Regularizer) *LDepthwiseConv2D {
	l.depthwiseRegularizer = depthwiseRegularizer
	return l
}

func (l *LDepthwiseConv2D) SetDilationRate(dilationRate []interface{}) *LDepthwiseConv2D {
	l.dilationRate = dilationRate
	return l
}

func (l *LDepthwiseConv2D) SetDtype(dtype DataType) *LDepthwiseConv2D {
	l.dtype = dtype
	return l
}

func (l *LDepthwiseConv2D) SetGroups(groups float64) *LDepthwiseConv2D {
	l.groups = groups
	return l
}

func (l *LDepthwiseConv2D) SetName(name string) *LDepthwiseConv2D {
	l.name = name
	return l
}

func (l *LDepthwiseConv2D) SetPadding(padding string) *LDepthwiseConv2D {
	l.padding = padding
	return l
}

func (l *LDepthwiseConv2D) SetShape(shape tf.Shape) *LDepthwiseConv2D {
	l.shape = shape
	return l
}

func (l *LDepthwiseConv2D) SetStrides(strides []interface{}) *LDepthwiseConv2D {
	l.strides = strides
	return l
}

func (l *LDepthwiseConv2D) SetTrainable(trainable bool) *LDepthwiseConv2D {
	l.trainable = trainable
	return l
}

func (l *LDepthwiseConv2D) SetUseBias(useBias bool) *LDepthwiseConv2D {
	l.useBias = useBias
	return l
}

func (l *LDepthwiseConv2D) SetLayerWeights(layerWeights interface{}) *LDepthwiseConv2D {
	l.layerWeights = layerWeights
	return l
}

func (l *LDepthwiseConv2D) GetShape() tf.Shape {
	return l.shape
}

func (l *LDepthwiseConv2D) GetDtype() DataType {
	return l.dtype
}

func (l *LDepthwiseConv2D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LDepthwiseConv2D) GetInputs() []Layer {
	return l.inputs
}

func (l *LDepthwiseConv2D) GetName() string {
	return l.name
}

func (l *LDepthwiseConv2D) GetLayerWeights() interface{} {
	return l.layerWeights
}

type jsonConfigLDepthwiseConv2D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LDepthwiseConv2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLDepthwiseConv2D{
		ClassName: "DepthwiseConv2D",
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
			"groups":                l.groups,
			"kernel_size":           l.kernelSize,
			"name":                  l.name,
			"padding":               l.padding,
			"strides":               l.strides,
			"trainable":             l.trainable,
			"use_bias":              l.useBias,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LDepthwiseConv2D) GetCustomLayerDefinition() string {
	return ``
}

package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LConv1D struct {
	activation          string
	activityRegularizer regularizer.Regularizer
	biasConstraint      constraint.Constraint
	biasInitializer     initializer.Initializer
	biasRegularizer     regularizer.Regularizer
	dataFormat          string
	dilationRate        float64
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
	strides             float64
	trainable           bool
	useBias             bool
}

func Conv1D(filters float64, kernelSize float64) *LConv1D {
	return &LConv1D{
		activation:          "linear",
		activityRegularizer: &regularizer.NilRegularizer{},
		biasConstraint:      &constraint.NilConstraint{},
		biasInitializer:     initializer.Zeros(),
		biasRegularizer:     &regularizer.NilRegularizer{},
		dataFormat:          "channels_last",
		dilationRate:        1,
		dtype:               Float32,
		filters:             filters,
		groups:              1,
		kernelConstraint:    &constraint.NilConstraint{},
		kernelInitializer:   initializer.GlorotUniform(),
		kernelRegularizer:   &regularizer.NilRegularizer{},
		kernelSize:          kernelSize,
		name:                UniqueName("conv1d"),
		padding:             "valid",
		strides:             1,
		trainable:           true,
		useBias:             true,
	}
}

func (l *LConv1D) SetActivation(activation string) *LConv1D {
	l.activation = activation
	return l
}

func (l *LConv1D) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LConv1D {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LConv1D) SetBiasConstraint(biasConstraint constraint.Constraint) *LConv1D {
	l.biasConstraint = biasConstraint
	return l
}

func (l *LConv1D) SetBiasInitializer(biasInitializer initializer.Initializer) *LConv1D {
	l.biasInitializer = biasInitializer
	return l
}

func (l *LConv1D) SetBiasRegularizer(biasRegularizer regularizer.Regularizer) *LConv1D {
	l.biasRegularizer = biasRegularizer
	return l
}

func (l *LConv1D) SetDataFormat(dataFormat string) *LConv1D {
	l.dataFormat = dataFormat
	return l
}

func (l *LConv1D) SetDilationRate(dilationRate float64) *LConv1D {
	l.dilationRate = dilationRate
	return l
}

func (l *LConv1D) SetDtype(dtype DataType) *LConv1D {
	l.dtype = dtype
	return l
}

func (l *LConv1D) SetGroups(groups float64) *LConv1D {
	l.groups = groups
	return l
}

func (l *LConv1D) SetKernelConstraint(kernelConstraint constraint.Constraint) *LConv1D {
	l.kernelConstraint = kernelConstraint
	return l
}

func (l *LConv1D) SetKernelInitializer(kernelInitializer initializer.Initializer) *LConv1D {
	l.kernelInitializer = kernelInitializer
	return l
}

func (l *LConv1D) SetKernelRegularizer(kernelRegularizer regularizer.Regularizer) *LConv1D {
	l.kernelRegularizer = kernelRegularizer
	return l
}

func (l *LConv1D) SetName(name string) *LConv1D {
	l.name = name
	return l
}

func (l *LConv1D) SetPadding(padding string) *LConv1D {
	l.padding = padding
	return l
}

func (l *LConv1D) SetShape(shape tf.Shape) *LConv1D {
	l.shape = shape
	return l
}

func (l *LConv1D) SetStrides(strides float64) *LConv1D {
	l.strides = strides
	return l
}

func (l *LConv1D) SetTrainable(trainable bool) *LConv1D {
	l.trainable = trainable
	return l
}

func (l *LConv1D) SetUseBias(useBias bool) *LConv1D {
	l.useBias = useBias
	return l
}

func (l *LConv1D) GetShape() tf.Shape {
	return l.shape
}

func (l *LConv1D) GetDtype() DataType {
	return l.dtype
}

func (l *LConv1D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LConv1D) GetInputs() []Layer {
	return l.inputs
}

func (l *LConv1D) GetName() string {
	return l.name
}

type jsonConfigLConv1D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LConv1D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLConv1D{
		ClassName: "Conv1D",
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

func (l *LConv1D) GetCustomLayerDefinition() string {
	return ``
}

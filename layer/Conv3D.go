package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LConv3D struct {
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
	layerWeights        []*tf.Tensor
}

func Conv3D(filters float64, kernelSize float64) *LConv3D {
	return &LConv3D{
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
		name:                UniqueName("conv3d"),
		padding:             "valid",
		strides:             []interface{}{1, 1, 1},
		trainable:           true,
		useBias:             true,
	}
}

func (l *LConv3D) SetActivation(activation string) *LConv3D {
	l.activation = activation
	return l
}

func (l *LConv3D) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LConv3D {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LConv3D) SetBiasConstraint(biasConstraint constraint.Constraint) *LConv3D {
	l.biasConstraint = biasConstraint
	return l
}

func (l *LConv3D) SetBiasInitializer(biasInitializer initializer.Initializer) *LConv3D {
	l.biasInitializer = biasInitializer
	return l
}

func (l *LConv3D) SetBiasRegularizer(biasRegularizer regularizer.Regularizer) *LConv3D {
	l.biasRegularizer = biasRegularizer
	return l
}

func (l *LConv3D) SetDataFormat(dataFormat interface{}) *LConv3D {
	l.dataFormat = dataFormat
	return l
}

func (l *LConv3D) SetDilationRate(dilationRate []interface{}) *LConv3D {
	l.dilationRate = dilationRate
	return l
}

func (l *LConv3D) SetDtype(dtype DataType) *LConv3D {
	l.dtype = dtype
	return l
}

func (l *LConv3D) SetGroups(groups float64) *LConv3D {
	l.groups = groups
	return l
}

func (l *LConv3D) SetKernelConstraint(kernelConstraint constraint.Constraint) *LConv3D {
	l.kernelConstraint = kernelConstraint
	return l
}

func (l *LConv3D) SetKernelInitializer(kernelInitializer initializer.Initializer) *LConv3D {
	l.kernelInitializer = kernelInitializer
	return l
}

func (l *LConv3D) SetKernelRegularizer(kernelRegularizer regularizer.Regularizer) *LConv3D {
	l.kernelRegularizer = kernelRegularizer
	return l
}

func (l *LConv3D) SetName(name string) *LConv3D {
	l.name = name
	return l
}

func (l *LConv3D) SetPadding(padding string) *LConv3D {
	l.padding = padding
	return l
}

func (l *LConv3D) SetShape(shape tf.Shape) *LConv3D {
	l.shape = shape
	return l
}

func (l *LConv3D) SetStrides(strides []interface{}) *LConv3D {
	l.strides = strides
	return l
}

func (l *LConv3D) SetTrainable(trainable bool) *LConv3D {
	l.trainable = trainable
	return l
}

func (l *LConv3D) SetUseBias(useBias bool) *LConv3D {
	l.useBias = useBias
	return l
}

func (l *LConv3D) SetLayerWeights(layerWeights []*tf.Tensor) *LConv3D {
	l.layerWeights = layerWeights
	return l
}

func (l *LConv3D) GetShape() tf.Shape {
	return l.shape
}

func (l *LConv3D) GetDtype() DataType {
	return l.dtype
}

func (l *LConv3D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LConv3D) GetInputs() []Layer {
	return l.inputs
}

func (l *LConv3D) GetName() string {
	return l.name
}

func (l *LConv3D) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLConv3D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LConv3D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLConv3D{
		ClassName: "Conv3D",
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

func (l *LConv3D) GetCustomLayerDefinition() string {
	return ``
}

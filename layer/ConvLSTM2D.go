package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LConvLSTM2D struct {
	activation           string
	activityRegularizer  regularizer.Regularizer
	biasConstraint       constraint.Constraint
	biasInitializer      initializer.Initializer
	biasRegularizer      regularizer.Regularizer
	dataFormat           interface{}
	dilationRate         []interface{}
	dropout              float64
	dtype                DataType
	filters              float64
	goBackwards          bool
	inputs               []Layer
	kernelConstraint     constraint.Constraint
	kernelInitializer    initializer.Initializer
	kernelRegularizer    regularizer.Regularizer
	kernelSize           float64
	name                 string
	padding              string
	recurrentActivation  string
	recurrentConstraint  constraint.Constraint
	recurrentDropout     float64
	recurrentInitializer initializer.Initializer
	recurrentRegularizer regularizer.Regularizer
	returnSequences      bool
	returnState          bool
	shape                tf.Shape
	stateful             bool
	strides              []interface{}
	timeMajor            bool
	trainable            bool
	unitForgetBias       bool
	unroll               bool
	useBias              bool
	layerWeights         []*tf.Tensor
}

func ConvLSTM2D(filters float64, kernelSize float64) *LConvLSTM2D {
	return &LConvLSTM2D{
		activation:           "tanh",
		activityRegularizer:  &regularizer.NilRegularizer{},
		biasConstraint:       &constraint.NilConstraint{},
		biasInitializer:      initializer.Zeros(),
		biasRegularizer:      &regularizer.NilRegularizer{},
		dataFormat:           nil,
		dilationRate:         []interface{}{1, 1},
		dropout:              0,
		dtype:                Float32,
		filters:              filters,
		goBackwards:          false,
		kernelConstraint:     &constraint.NilConstraint{},
		kernelInitializer:    initializer.GlorotUniform(),
		kernelRegularizer:    &regularizer.NilRegularizer{},
		kernelSize:           kernelSize,
		name:                 UniqueName("conv_lst_m2d"),
		padding:              "valid",
		recurrentActivation:  "hard_sigmoid",
		recurrentConstraint:  &constraint.NilConstraint{},
		recurrentDropout:     0,
		recurrentInitializer: initializer.Orthogonal(),
		recurrentRegularizer: &regularizer.NilRegularizer{},
		returnSequences:      false,
		returnState:          false,
		stateful:             false,
		strides:              []interface{}{1, 1},
		timeMajor:            false,
		trainable:            true,
		unitForgetBias:       true,
		unroll:               false,
		useBias:              true,
	}
}

func (l *LConvLSTM2D) SetActivation(activation string) *LConvLSTM2D {
	l.activation = activation
	return l
}

func (l *LConvLSTM2D) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LConvLSTM2D {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LConvLSTM2D) SetBiasConstraint(biasConstraint constraint.Constraint) *LConvLSTM2D {
	l.biasConstraint = biasConstraint
	return l
}

func (l *LConvLSTM2D) SetBiasInitializer(biasInitializer initializer.Initializer) *LConvLSTM2D {
	l.biasInitializer = biasInitializer
	return l
}

func (l *LConvLSTM2D) SetBiasRegularizer(biasRegularizer regularizer.Regularizer) *LConvLSTM2D {
	l.biasRegularizer = biasRegularizer
	return l
}

func (l *LConvLSTM2D) SetDataFormat(dataFormat interface{}) *LConvLSTM2D {
	l.dataFormat = dataFormat
	return l
}

func (l *LConvLSTM2D) SetDilationRate(dilationRate []interface{}) *LConvLSTM2D {
	l.dilationRate = dilationRate
	return l
}

func (l *LConvLSTM2D) SetDropout(dropout float64) *LConvLSTM2D {
	l.dropout = dropout
	return l
}

func (l *LConvLSTM2D) SetDtype(dtype DataType) *LConvLSTM2D {
	l.dtype = dtype
	return l
}

func (l *LConvLSTM2D) SetGoBackwards(goBackwards bool) *LConvLSTM2D {
	l.goBackwards = goBackwards
	return l
}

func (l *LConvLSTM2D) SetKernelConstraint(kernelConstraint constraint.Constraint) *LConvLSTM2D {
	l.kernelConstraint = kernelConstraint
	return l
}

func (l *LConvLSTM2D) SetKernelInitializer(kernelInitializer initializer.Initializer) *LConvLSTM2D {
	l.kernelInitializer = kernelInitializer
	return l
}

func (l *LConvLSTM2D) SetKernelRegularizer(kernelRegularizer regularizer.Regularizer) *LConvLSTM2D {
	l.kernelRegularizer = kernelRegularizer
	return l
}

func (l *LConvLSTM2D) SetName(name string) *LConvLSTM2D {
	l.name = name
	return l
}

func (l *LConvLSTM2D) SetPadding(padding string) *LConvLSTM2D {
	l.padding = padding
	return l
}

func (l *LConvLSTM2D) SetRecurrentActivation(recurrentActivation string) *LConvLSTM2D {
	l.recurrentActivation = recurrentActivation
	return l
}

func (l *LConvLSTM2D) SetRecurrentConstraint(recurrentConstraint constraint.Constraint) *LConvLSTM2D {
	l.recurrentConstraint = recurrentConstraint
	return l
}

func (l *LConvLSTM2D) SetRecurrentDropout(recurrentDropout float64) *LConvLSTM2D {
	l.recurrentDropout = recurrentDropout
	return l
}

func (l *LConvLSTM2D) SetRecurrentInitializer(recurrentInitializer initializer.Initializer) *LConvLSTM2D {
	l.recurrentInitializer = recurrentInitializer
	return l
}

func (l *LConvLSTM2D) SetRecurrentRegularizer(recurrentRegularizer regularizer.Regularizer) *LConvLSTM2D {
	l.recurrentRegularizer = recurrentRegularizer
	return l
}

func (l *LConvLSTM2D) SetReturnSequences(returnSequences bool) *LConvLSTM2D {
	l.returnSequences = returnSequences
	return l
}

func (l *LConvLSTM2D) SetReturnState(returnState bool) *LConvLSTM2D {
	l.returnState = returnState
	return l
}

func (l *LConvLSTM2D) SetShape(shape tf.Shape) *LConvLSTM2D {
	l.shape = shape
	return l
}

func (l *LConvLSTM2D) SetStateful(stateful bool) *LConvLSTM2D {
	l.stateful = stateful
	return l
}

func (l *LConvLSTM2D) SetStrides(strides []interface{}) *LConvLSTM2D {
	l.strides = strides
	return l
}

func (l *LConvLSTM2D) SetTimeMajor(timeMajor bool) *LConvLSTM2D {
	l.timeMajor = timeMajor
	return l
}

func (l *LConvLSTM2D) SetTrainable(trainable bool) *LConvLSTM2D {
	l.trainable = trainable
	return l
}

func (l *LConvLSTM2D) SetUnitForgetBias(unitForgetBias bool) *LConvLSTM2D {
	l.unitForgetBias = unitForgetBias
	return l
}

func (l *LConvLSTM2D) SetUnroll(unroll bool) *LConvLSTM2D {
	l.unroll = unroll
	return l
}

func (l *LConvLSTM2D) SetUseBias(useBias bool) *LConvLSTM2D {
	l.useBias = useBias
	return l
}

func (l *LConvLSTM2D) SetLayerWeights(layerWeights []*tf.Tensor) *LConvLSTM2D {
	l.layerWeights = layerWeights
	return l
}

func (l *LConvLSTM2D) GetShape() tf.Shape {
	return l.shape
}

func (l *LConvLSTM2D) GetDtype() DataType {
	return l.dtype
}

func (l *LConvLSTM2D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LConvLSTM2D) GetInputs() []Layer {
	return l.inputs
}

func (l *LConvLSTM2D) GetName() string {
	return l.name
}

func (l *LConvLSTM2D) GetLayerWeights() []*tf.Tensor {
	return l.layerWeights
}

type jsonConfigLConvLSTM2D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LConvLSTM2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLConvLSTM2D{
		ClassName: "ConvLSTM2D",
		Name:      l.name,
		Config: map[string]interface{}{
			"activation":            l.activation,
			"activity_regularizer":  l.activityRegularizer.GetKerasLayerConfig(),
			"bias_constraint":       l.biasConstraint.GetKerasLayerConfig(),
			"bias_initializer":      l.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer":      l.biasRegularizer.GetKerasLayerConfig(),
			"data_format":           l.dataFormat,
			"dilation_rate":         l.dilationRate,
			"dropout":               l.dropout,
			"dtype":                 l.dtype.String(),
			"filters":               l.filters,
			"go_backwards":          l.goBackwards,
			"kernel_constraint":     l.kernelConstraint.GetKerasLayerConfig(),
			"kernel_initializer":    l.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer":    l.kernelRegularizer.GetKerasLayerConfig(),
			"kernel_size":           l.kernelSize,
			"name":                  l.name,
			"padding":               l.padding,
			"recurrent_activation":  l.recurrentActivation,
			"recurrent_constraint":  l.recurrentConstraint.GetKerasLayerConfig(),
			"recurrent_dropout":     l.recurrentDropout,
			"recurrent_initializer": l.recurrentInitializer.GetKerasLayerConfig(),
			"recurrent_regularizer": l.recurrentRegularizer.GetKerasLayerConfig(),
			"return_sequences":      l.returnSequences,
			"return_state":          l.returnState,
			"stateful":              l.stateful,
			"strides":               l.strides,
			"time_major":            l.timeMajor,
			"trainable":             l.trainable,
			"unit_forget_bias":      l.unitForgetBias,
			"unroll":                l.unroll,
			"use_bias":              l.useBias,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LConvLSTM2D) GetCustomLayerDefinition() string {
	return ``
}

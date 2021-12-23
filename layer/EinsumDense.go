package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type EinsumDense struct {
	name                string
	dtype               DataType
	inputs              []Layer
	shape               tf.Shape
	trainable           bool
	equation            float64
	outputShape         float64
	activation          string
	biasAxes            interface{}
	kernelInitializer   initializer.Initializer
	biasInitializer     initializer.Initializer
	kernelRegularizer   regularizer.Regularizer
	biasRegularizer     regularizer.Regularizer
	activityRegularizer regularizer.Regularizer
	kernelConstraint    constraint.Constraint
	biasConstraint      constraint.Constraint
}

func NewEinsumDense(equation float64, outputShape float64, options ...EinsumDenseOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		e := &EinsumDense{
			equation:            equation,
			outputShape:         outputShape,
			activation:          "linear",
			biasAxes:            nil,
			kernelInitializer:   &initializer.GlorotUniform{},
			biasInitializer:     &initializer.Zeros{},
			kernelRegularizer:   &regularizer.NilRegularizer{},
			biasRegularizer:     &regularizer.NilRegularizer{},
			activityRegularizer: &regularizer.NilRegularizer{},
			kernelConstraint:    &constraint.NilConstraint{},
			biasConstraint:      &constraint.NilConstraint{},
			trainable:           true,
			inputs:              inputs,
			name:                UniqueName("einsumdense"),
		}
		for _, option := range options {
			option(e)
		}
		return e
	}
}

type EinsumDenseOption func(*EinsumDense)

func EinsumDenseWithName(name string) func(e *EinsumDense) {
	return func(e *EinsumDense) {
		e.name = name
	}
}

func EinsumDenseWithDtype(dtype DataType) func(e *EinsumDense) {
	return func(e *EinsumDense) {
		e.dtype = dtype
	}
}

func EinsumDenseWithTrainable(trainable bool) func(e *EinsumDense) {
	return func(e *EinsumDense) {
		e.trainable = trainable
	}
}

func EinsumDenseWithActivation(activation string) func(e *EinsumDense) {
	return func(e *EinsumDense) {
		e.activation = activation
	}
}

func EinsumDenseWithBiasAxes(biasAxes interface{}) func(e *EinsumDense) {
	return func(e *EinsumDense) {
		e.biasAxes = biasAxes
	}
}

func EinsumDenseWithKernelInitializer(kernelInitializer initializer.Initializer) func(e *EinsumDense) {
	return func(e *EinsumDense) {
		e.kernelInitializer = kernelInitializer
	}
}

func EinsumDenseWithBiasInitializer(biasInitializer initializer.Initializer) func(e *EinsumDense) {
	return func(e *EinsumDense) {
		e.biasInitializer = biasInitializer
	}
}

func EinsumDenseWithKernelRegularizer(kernelRegularizer regularizer.Regularizer) func(e *EinsumDense) {
	return func(e *EinsumDense) {
		e.kernelRegularizer = kernelRegularizer
	}
}

func EinsumDenseWithBiasRegularizer(biasRegularizer regularizer.Regularizer) func(e *EinsumDense) {
	return func(e *EinsumDense) {
		e.biasRegularizer = biasRegularizer
	}
}

func EinsumDenseWithActivityRegularizer(activityRegularizer regularizer.Regularizer) func(e *EinsumDense) {
	return func(e *EinsumDense) {
		e.activityRegularizer = activityRegularizer
	}
}

func EinsumDenseWithKernelConstraint(kernelConstraint constraint.Constraint) func(e *EinsumDense) {
	return func(e *EinsumDense) {
		e.kernelConstraint = kernelConstraint
	}
}

func EinsumDenseWithBiasConstraint(biasConstraint constraint.Constraint) func(e *EinsumDense) {
	return func(e *EinsumDense) {
		e.biasConstraint = biasConstraint
	}
}

func (e *EinsumDense) GetShape() tf.Shape {
	return e.shape
}

func (e *EinsumDense) GetDtype() DataType {
	return e.dtype
}

func (e *EinsumDense) SetInput(inputs []Layer) {
	e.inputs = inputs
	e.dtype = inputs[0].GetDtype()
}

func (e *EinsumDense) GetInputs() []Layer {
	return e.inputs
}

func (e *EinsumDense) GetName() string {
	return e.name
}

type jsonConfigEinsumDense struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (e *EinsumDense) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range e.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigEinsumDense{
		ClassName: "EinsumDense",
		Name:      e.name,
		Config: map[string]interface{}{
			"activation":           e.activation,
			"activity_regularizer": e.activityRegularizer.GetKerasLayerConfig(),
			"bias_axes":            e.biasAxes,
			"bias_constraint":      e.biasConstraint.GetKerasLayerConfig(),
			"bias_initializer":     e.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer":     e.biasRegularizer.GetKerasLayerConfig(),
			"dtype":                e.dtype.String(),
			"equation":             e.equation,
			"kernel_constraint":    e.kernelConstraint.GetKerasLayerConfig(),
			"kernel_initializer":   e.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer":   e.kernelRegularizer.GetKerasLayerConfig(),
			"name":                 e.name,
			"output_shape":         e.outputShape,
			"trainable":            e.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (e *EinsumDense) GetCustomLayerDefinition() string {
	return ``
}

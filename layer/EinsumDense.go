package layer

import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"
import tf "github.com/galeone/tensorflow/tensorflow/go"

type LEinsumDense struct {
	activation          string
	activityRegularizer regularizer.Regularizer
	biasAxes            interface{}
	biasConstraint      constraint.Constraint
	biasInitializer     initializer.Initializer
	biasRegularizer     regularizer.Regularizer
	dtype               DataType
	equation            float64
	inputs              []Layer
	kernelConstraint    constraint.Constraint
	kernelInitializer   initializer.Initializer
	kernelRegularizer   regularizer.Regularizer
	name                string
	outputShape         float64
	shape               tf.Shape
	trainable           bool
}

func EinsumDense(equation float64, outputShape float64) *LEinsumDense {
	return &LEinsumDense{
		activation:          "linear",
		activityRegularizer: &regularizer.NilRegularizer{},
		biasAxes:            nil,
		biasConstraint:      &constraint.NilConstraint{},
		biasInitializer:     initializer.Zeros(),
		biasRegularizer:     &regularizer.NilRegularizer{},
		dtype:               Float32,
		equation:            equation,
		kernelConstraint:    &constraint.NilConstraint{},
		kernelInitializer:   initializer.GlorotUniform(),
		kernelRegularizer:   &regularizer.NilRegularizer{},
		name:                UniqueName("einsum_dense"),
		outputShape:         outputShape,
		trainable:           true,
	}
}

func (l *LEinsumDense) SetActivation(activation string) *LEinsumDense {
	l.activation = activation
	return l
}

func (l *LEinsumDense) SetActivityRegularizer(activityRegularizer regularizer.Regularizer) *LEinsumDense {
	l.activityRegularizer = activityRegularizer
	return l
}

func (l *LEinsumDense) SetBiasAxes(biasAxes interface{}) *LEinsumDense {
	l.biasAxes = biasAxes
	return l
}

func (l *LEinsumDense) SetBiasConstraint(biasConstraint constraint.Constraint) *LEinsumDense {
	l.biasConstraint = biasConstraint
	return l
}

func (l *LEinsumDense) SetBiasInitializer(biasInitializer initializer.Initializer) *LEinsumDense {
	l.biasInitializer = biasInitializer
	return l
}

func (l *LEinsumDense) SetBiasRegularizer(biasRegularizer regularizer.Regularizer) *LEinsumDense {
	l.biasRegularizer = biasRegularizer
	return l
}

func (l *LEinsumDense) SetDtype(dtype DataType) *LEinsumDense {
	l.dtype = dtype
	return l
}

func (l *LEinsumDense) SetKernelConstraint(kernelConstraint constraint.Constraint) *LEinsumDense {
	l.kernelConstraint = kernelConstraint
	return l
}

func (l *LEinsumDense) SetKernelInitializer(kernelInitializer initializer.Initializer) *LEinsumDense {
	l.kernelInitializer = kernelInitializer
	return l
}

func (l *LEinsumDense) SetKernelRegularizer(kernelRegularizer regularizer.Regularizer) *LEinsumDense {
	l.kernelRegularizer = kernelRegularizer
	return l
}

func (l *LEinsumDense) SetName(name string) *LEinsumDense {
	l.name = name
	return l
}

func (l *LEinsumDense) SetShape(shape tf.Shape) *LEinsumDense {
	l.shape = shape
	return l
}

func (l *LEinsumDense) SetTrainable(trainable bool) *LEinsumDense {
	l.trainable = trainable
	return l
}

func (l *LEinsumDense) GetShape() tf.Shape {
	return l.shape
}

func (l *LEinsumDense) GetDtype() DataType {
	return l.dtype
}

func (l *LEinsumDense) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LEinsumDense) GetInputs() []Layer {
	return l.inputs
}

func (l *LEinsumDense) GetName() string {
	return l.name
}

type jsonConfigLEinsumDense struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LEinsumDense) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLEinsumDense{
		ClassName: "EinsumDense",
		Name:      l.name,
		Config: map[string]interface{}{
			"activation":           l.activation,
			"activity_regularizer": l.activityRegularizer.GetKerasLayerConfig(),
			"bias_axes":            l.biasAxes,
			"bias_constraint":      l.biasConstraint.GetKerasLayerConfig(),
			"bias_initializer":     l.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer":     l.biasRegularizer.GetKerasLayerConfig(),
			"dtype":                l.dtype.String(),
			"equation":             l.equation,
			"kernel_constraint":    l.kernelConstraint.GetKerasLayerConfig(),
			"kernel_initializer":   l.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer":   l.kernelRegularizer.GetKerasLayerConfig(),
			"name":                 l.name,
			"output_shape":         l.outputShape,
			"trainable":            l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LEinsumDense) GetCustomLayerDefinition() string {
	return ``
}

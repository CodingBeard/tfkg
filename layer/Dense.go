package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"
import "github.com/codingbeard/tfkg/layer/constraint"
import "github.com/codingbeard/tfkg/layer/initializer"
import "github.com/codingbeard/tfkg/layer/regularizer"

type Dense struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	units float64
	activation string
	useBias bool
	kernelInitializer initializer.Initializer
	biasInitializer initializer.Initializer
	kernelRegularizer regularizer.Regularizer
	biasRegularizer regularizer.Regularizer
	activityRegularizer regularizer.Regularizer
	kernelConstraint constraint.Constraint
	biasConstraint constraint.Constraint
}

func NewDense(units float64, options ...DenseOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		d := &Dense{
			units: units,
			activation: "linear",
			useBias: true,
			kernelInitializer: &initializer.GlorotUniform{},
			biasInitializer: &initializer.Zeros{},
			kernelRegularizer: &regularizer.NilRegularizer{},
			biasRegularizer: &regularizer.NilRegularizer{},
			activityRegularizer: &regularizer.NilRegularizer{},
			kernelConstraint: &constraint.NilConstraint{},
			biasConstraint: &constraint.NilConstraint{},
			trainable: true,
			inputs: inputs,
			name: uniqueName("dense"),		
		}
		for _, option := range options {
			option(d)
		}
		return d
	}
}

type DenseOption func (*Dense)

func DenseWithName(name string) func(d *Dense) {
	 return func(d *Dense) {
		d.name = name
	}
}

func DenseWithDtype(dtype DataType) func(d *Dense) {
	 return func(d *Dense) {
		d.dtype = dtype
	}
}

func DenseWithTrainable(trainable bool) func(d *Dense) {
	 return func(d *Dense) {
		d.trainable = trainable
	}
}

func DenseWithActivation(activation string) func(d *Dense) {
	 return func(d *Dense) {
		d.activation = activation
	}
}

func DenseWithUseBias(useBias bool) func(d *Dense) {
	 return func(d *Dense) {
		d.useBias = useBias
	}
}

func DenseWithKernelInitializer(kernelInitializer initializer.Initializer) func(d *Dense) {
	 return func(d *Dense) {
		d.kernelInitializer = kernelInitializer
	}
}

func DenseWithBiasInitializer(biasInitializer initializer.Initializer) func(d *Dense) {
	 return func(d *Dense) {
		d.biasInitializer = biasInitializer
	}
}

func DenseWithKernelRegularizer(kernelRegularizer regularizer.Regularizer) func(d *Dense) {
	 return func(d *Dense) {
		d.kernelRegularizer = kernelRegularizer
	}
}

func DenseWithBiasRegularizer(biasRegularizer regularizer.Regularizer) func(d *Dense) {
	 return func(d *Dense) {
		d.biasRegularizer = biasRegularizer
	}
}

func DenseWithActivityRegularizer(activityRegularizer regularizer.Regularizer) func(d *Dense) {
	 return func(d *Dense) {
		d.activityRegularizer = activityRegularizer
	}
}

func DenseWithKernelConstraint(kernelConstraint constraint.Constraint) func(d *Dense) {
	 return func(d *Dense) {
		d.kernelConstraint = kernelConstraint
	}
}

func DenseWithBiasConstraint(biasConstraint constraint.Constraint) func(d *Dense) {
	 return func(d *Dense) {
		d.biasConstraint = biasConstraint
	}
}


func (d *Dense) GetShape() tf.Shape {
	return d.shape
}

func (d *Dense) GetDtype() DataType {
	return d.dtype
}

func (d *Dense) SetInput(inputs []Layer) {
	d.inputs = inputs
	d.dtype = inputs[0].GetDtype()
}

func (d *Dense) GetInputs() []Layer {
	return d.inputs
}

func (d *Dense) GetName() string {
	return d.name
}


type jsonConfigDense struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (d *Dense) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range d.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigDense{
		ClassName: "Dense",
		Name: d.name,
		Config: map[string]interface{}{
			"kernel_initializer": d.kernelInitializer.GetKerasLayerConfig(),
			"kernel_regularizer": d.kernelRegularizer.GetKerasLayerConfig(),
			"activity_regularizer": d.activityRegularizer.GetKerasLayerConfig(),
			"kernel_constraint": d.kernelConstraint.GetKerasLayerConfig(),
			"name": d.name,
			"dtype": d.dtype.String(),
			"units": d.units,
			"bias_initializer": d.biasInitializer.GetKerasLayerConfig(),
			"bias_regularizer": d.biasRegularizer.GetKerasLayerConfig(),
			"bias_constraint": d.biasConstraint.GetKerasLayerConfig(),
			"trainable": d.trainable,
			"activation": d.activation,
			"use_bias": d.useBias,
		},
		InboundNodes: inboundNodes,
	}
}
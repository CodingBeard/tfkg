package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type Dropout struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	rate float64
	noiseShape interface{}
	seed interface{}
}

func NewDropout(rate float64, options ...DropoutOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		d := &Dropout{
			rate: rate,
			noiseShape: nil,
			seed: nil,
			trainable: true,
			inputs: inputs,
			name: uniqueName("dropout"),		
		}
		for _, option := range options {
			option(d)
		}
		return d
	}
}

type DropoutOption func (*Dropout)

func DropoutWithName(name string) func(d *Dropout) {
	 return func(d *Dropout) {
		d.name = name
	}
}

func DropoutWithDtype(dtype DataType) func(d *Dropout) {
	 return func(d *Dropout) {
		d.dtype = dtype
	}
}

func DropoutWithTrainable(trainable bool) func(d *Dropout) {
	 return func(d *Dropout) {
		d.trainable = trainable
	}
}

func DropoutWithNoiseShape(noiseShape interface{}) func(d *Dropout) {
	 return func(d *Dropout) {
		d.noiseShape = noiseShape
	}
}

func DropoutWithSeed(seed interface{}) func(d *Dropout) {
	 return func(d *Dropout) {
		d.seed = seed
	}
}


func (d *Dropout) GetShape() tf.Shape {
	return d.shape
}

func (d *Dropout) GetDtype() DataType {
	return d.dtype
}

func (d *Dropout) SetInput(inputs []Layer) {
	d.inputs = inputs
	d.dtype = inputs[0].GetDtype()
}

func (d *Dropout) GetInputs() []Layer {
	return d.inputs
}

func (d *Dropout) GetName() string {
	return d.name
}


type jsonConfigDropout struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (d *Dropout) GetKerasLayerConfig() interface{} {
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
	return jsonConfigDropout{
		ClassName: "Dropout",
		Name: d.name,
		Config: map[string]interface{}{
			"dtype": d.dtype.String(),
			"rate": d.rate,
			"noise_shape": d.noiseShape,
			"seed": d.seed,
			"name": d.name,
			"trainable": d.trainable,
		},
		InboundNodes: inboundNodes,
	}
}
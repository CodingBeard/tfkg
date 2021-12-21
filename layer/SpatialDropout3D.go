package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type SpatialDropout3D struct {
	name string
	dtype DataType
	inputs []Layer
	shape tf.Shape
	trainable bool
	rate float64
	dataFormat interface{}
	noiseShape interface{}
	seed interface{}
}

func NewSpatialDropout3D(rate float64, options ...SpatialDropout3DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		s := &SpatialDropout3D{
			rate: rate,
			dataFormat: nil,
			noiseShape: nil,
			seed: nil,
			trainable: true,
			inputs: inputs,
			name: uniqueName("spatialdropout3d"),		
		}
		for _, option := range options {
			option(s)
		}
		return s
	}
}

type SpatialDropout3DOption func (*SpatialDropout3D)

func SpatialDropout3DWithName(name string) func(s *SpatialDropout3D) {
	 return func(s *SpatialDropout3D) {
		s.name = name
	}
}

func SpatialDropout3DWithDtype(dtype DataType) func(s *SpatialDropout3D) {
	 return func(s *SpatialDropout3D) {
		s.dtype = dtype
	}
}

func SpatialDropout3DWithTrainable(trainable bool) func(s *SpatialDropout3D) {
	 return func(s *SpatialDropout3D) {
		s.trainable = trainable
	}
}

func SpatialDropout3DWithDataFormat(dataFormat interface{}) func(s *SpatialDropout3D) {
	 return func(s *SpatialDropout3D) {
		s.dataFormat = dataFormat
	}
}


func (s *SpatialDropout3D) GetShape() tf.Shape {
	return s.shape
}

func (s *SpatialDropout3D) GetDtype() DataType {
	return s.dtype
}

func (s *SpatialDropout3D) SetInput(inputs []Layer) {
	s.inputs = inputs
	s.dtype = inputs[0].GetDtype()
}

func (s *SpatialDropout3D) GetInputs() []Layer {
	return s.inputs
}

func (s *SpatialDropout3D) GetName() string {
	return s.name
}


type jsonConfigSpatialDropout3D struct {
	ClassName string `json:"class_name"`
	Name string `json:"name"`
	Config map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}
func (s *SpatialDropout3D) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range s.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigSpatialDropout3D{
		ClassName: "SpatialDropout3D",
		Name: s.name,
		Config: map[string]interface{}{
			"name": s.name,
			"trainable": s.trainable,
			"dtype": s.dtype.String(),
			"rate": s.rate,
			"noise_shape": s.noiseShape,
			"seed": s.seed,
		},
		InboundNodes: inboundNodes,
	}
}
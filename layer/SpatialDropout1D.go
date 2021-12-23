package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type SpatialDropout1D struct {
	name       string
	dtype      DataType
	inputs     []Layer
	shape      tf.Shape
	trainable  bool
	rate       float64
	noiseShape interface{}
	seed       interface{}
}

func NewSpatialDropout1D(rate float64, options ...SpatialDropout1DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		s := &SpatialDropout1D{
			rate:       rate,
			noiseShape: nil,
			seed:       nil,
			trainable:  true,
			inputs:     inputs,
			name:       UniqueName("spatialdropout1d"),
		}
		for _, option := range options {
			option(s)
		}
		return s
	}
}

type SpatialDropout1DOption func(*SpatialDropout1D)

func SpatialDropout1DWithName(name string) func(s *SpatialDropout1D) {
	return func(s *SpatialDropout1D) {
		s.name = name
	}
}

func SpatialDropout1DWithDtype(dtype DataType) func(s *SpatialDropout1D) {
	return func(s *SpatialDropout1D) {
		s.dtype = dtype
	}
}

func SpatialDropout1DWithTrainable(trainable bool) func(s *SpatialDropout1D) {
	return func(s *SpatialDropout1D) {
		s.trainable = trainable
	}
}

func (s *SpatialDropout1D) GetShape() tf.Shape {
	return s.shape
}

func (s *SpatialDropout1D) GetDtype() DataType {
	return s.dtype
}

func (s *SpatialDropout1D) SetInput(inputs []Layer) {
	s.inputs = inputs
	s.dtype = inputs[0].GetDtype()
}

func (s *SpatialDropout1D) GetInputs() []Layer {
	return s.inputs
}

func (s *SpatialDropout1D) GetName() string {
	return s.name
}

type jsonConfigSpatialDropout1D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (s *SpatialDropout1D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigSpatialDropout1D{
		ClassName: "SpatialDropout1D",
		Name:      s.name,
		Config: map[string]interface{}{
			"dtype":       s.dtype.String(),
			"name":        s.name,
			"noise_shape": s.noiseShape,
			"rate":        s.rate,
			"seed":        s.seed,
			"trainable":   s.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (s *SpatialDropout1D) GetCustomLayerDefinition() string {
	return ``
}

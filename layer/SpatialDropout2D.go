package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type SpatialDropout2D struct {
	name       string
	dtype      DataType
	inputs     []Layer
	shape      tf.Shape
	trainable  bool
	rate       float64
	dataFormat interface{}
	noiseShape interface{}
	seed       interface{}
}

func NewSpatialDropout2D(rate float64, options ...SpatialDropout2DOption) func(inputs ...Layer) Layer {
	return func(inputs ...Layer) Layer {
		s := &SpatialDropout2D{
			rate:       rate,
			dataFormat: nil,
			seed:       nil,
			noiseShape: nil,
			trainable:  true,
			inputs:     inputs,
			name:       UniqueName("spatialdropout2d"),
		}
		for _, option := range options {
			option(s)
		}
		return s
	}
}

type SpatialDropout2DOption func(*SpatialDropout2D)

func SpatialDropout2DWithName(name string) func(s *SpatialDropout2D) {
	return func(s *SpatialDropout2D) {
		s.name = name
	}
}

func SpatialDropout2DWithDtype(dtype DataType) func(s *SpatialDropout2D) {
	return func(s *SpatialDropout2D) {
		s.dtype = dtype
	}
}

func SpatialDropout2DWithTrainable(trainable bool) func(s *SpatialDropout2D) {
	return func(s *SpatialDropout2D) {
		s.trainable = trainable
	}
}

func SpatialDropout2DWithDataFormat(dataFormat interface{}) func(s *SpatialDropout2D) {
	return func(s *SpatialDropout2D) {
		s.dataFormat = dataFormat
	}
}

func (s *SpatialDropout2D) GetShape() tf.Shape {
	return s.shape
}

func (s *SpatialDropout2D) GetDtype() DataType {
	return s.dtype
}

func (s *SpatialDropout2D) SetInput(inputs []Layer) {
	s.inputs = inputs
	s.dtype = inputs[0].GetDtype()
}

func (s *SpatialDropout2D) GetInputs() []Layer {
	return s.inputs
}

func (s *SpatialDropout2D) GetName() string {
	return s.name
}

type jsonConfigSpatialDropout2D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (s *SpatialDropout2D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigSpatialDropout2D{
		ClassName: "SpatialDropout2D",
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

func (s *SpatialDropout2D) GetCustomLayerDefinition() string {
	return ``
}

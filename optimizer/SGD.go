package optimizer

type SGD struct {
	learningRate float64
	momentum     float64
	nesterov     bool
	name         string
	decay        float64
}

func NewSGD() *SGD {
	return &SGD{
		learningRate: 0.01,
		momentum:     0,
		nesterov:     false,
		name:         "SGD",
	}
}

func SGDWithLearningRate(learningRate float64) func(s *SGD) {
	return func(s *SGD) {
		s.learningRate = learningRate
	}
}

func SGDWithMomentum(momentum float64) func(s *SGD) {
	return func(s *SGD) {
		s.momentum = momentum
	}
}

func SGDWithNesterov(nesterov bool) func(s *SGD) {
	return func(s *SGD) {
		s.nesterov = nesterov
	}
}

func SGDWithName(name string) func(s *SGD) {
	return func(s *SGD) {
		s.name = name
	}
}

type jsonConfigSGD struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (s *SGD) GetKerasLayerConfig() interface{} {
	if s == nil {
		return nil
	}
	return jsonConfigSGD{
		ClassName: "SGD",
		Config: map[string]interface{}{
			"decay":         s.decay,
			"learning_rate": s.learningRate,
			"momentum":      s.momentum,
			"name":          s.name,
			"nesterov":      s.nesterov,
		},
	}
}

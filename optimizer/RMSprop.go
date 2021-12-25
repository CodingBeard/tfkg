package optimizer

type RMSprop struct {
	learningRate float64
	rho          float64
	momentum     float64
	epsilon      float64
	centered     bool
	name         string
	decay        float64
}

func NewRMSprop() *RMSprop {
	return &RMSprop{
		learningRate: 0.001,
		rho:          0.9,
		momentum:     0,
		epsilon:      1e-07,
		centered:     false,
		name:         "RMSprop",
	}
}

func RMSpropWithLearningRate(learningRate float64) func(r *RMSprop) {
	return func(r *RMSprop) {
		r.learningRate = learningRate
	}
}

func RMSpropWithRho(rho float64) func(r *RMSprop) {
	return func(r *RMSprop) {
		r.rho = rho
	}
}

func RMSpropWithMomentum(momentum float64) func(r *RMSprop) {
	return func(r *RMSprop) {
		r.momentum = momentum
	}
}

func RMSpropWithEpsilon(epsilon float64) func(r *RMSprop) {
	return func(r *RMSprop) {
		r.epsilon = epsilon
	}
}

func RMSpropWithCentered(centered bool) func(r *RMSprop) {
	return func(r *RMSprop) {
		r.centered = centered
	}
}

func RMSpropWithName(name string) func(r *RMSprop) {
	return func(r *RMSprop) {
		r.name = name
	}
}

type jsonConfigRMSprop struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (r *RMSprop) GetKerasLayerConfig() interface{} {
	if r == nil {
		return nil
	}
	return jsonConfigRMSprop{
		ClassName: "RMSprop",
		Config: map[string]interface{}{
			"centered":      r.centered,
			"decay":         r.decay,
			"epsilon":       r.epsilon,
			"learning_rate": r.learningRate,
			"momentum":      r.momentum,
			"name":          r.name,
			"rho":           r.rho,
		},
	}
}

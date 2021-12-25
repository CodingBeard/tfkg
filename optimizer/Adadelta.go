package optimizer

type Adadelta struct {
	learningRate float64
	rho          float64
	epsilon      float64
	name         string
	decay        float64
}

func NewAdadelta() *Adadelta {
	return &Adadelta{
		learningRate: 0.001,
		rho:          0.95,
		epsilon:      1e-07,
		name:         "Adadelta",
	}
}

func AdadeltaWithLearningRate(learningRate float64) func(a *Adadelta) {
	return func(a *Adadelta) {
		a.learningRate = learningRate
	}
}

func AdadeltaWithRho(rho float64) func(a *Adadelta) {
	return func(a *Adadelta) {
		a.rho = rho
	}
}

func AdadeltaWithEpsilon(epsilon float64) func(a *Adadelta) {
	return func(a *Adadelta) {
		a.epsilon = epsilon
	}
}

func AdadeltaWithName(name string) func(a *Adadelta) {
	return func(a *Adadelta) {
		a.name = name
	}
}

type jsonConfigAdadelta struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (a *Adadelta) GetKerasLayerConfig() interface{} {
	if a == nil {
		return nil
	}
	return jsonConfigAdadelta{
		ClassName: "Adadelta",
		Config: map[string]interface{}{
			"decay":         a.decay,
			"epsilon":       a.epsilon,
			"learning_rate": a.learningRate,
			"name":          a.name,
			"rho":           a.rho,
		},
	}
}

package optimizer

type Adam struct {
	learningRate float64
	beta1        float64
	beta2        float64
	epsilon      float64
	amsgrad      bool
	name         string
	decay        float64
}

func NewAdam() *Adam {
	return &Adam{
		learningRate: 0.001,
		beta1:        0.9,
		beta2:        0.999,
		epsilon:      1e-07,
		amsgrad:      false,
		name:         "Adam",
	}
}

func AdamWithLearningRate(learningRate float64) func(a *Adam) {
	return func(a *Adam) {
		a.learningRate = learningRate
	}
}

func AdamWithBeta1(beta1 float64) func(a *Adam) {
	return func(a *Adam) {
		a.beta1 = beta1
	}
}

func AdamWithBeta2(beta2 float64) func(a *Adam) {
	return func(a *Adam) {
		a.beta2 = beta2
	}
}

func AdamWithEpsilon(epsilon float64) func(a *Adam) {
	return func(a *Adam) {
		a.epsilon = epsilon
	}
}

func AdamWithAmsgrad(amsgrad bool) func(a *Adam) {
	return func(a *Adam) {
		a.amsgrad = amsgrad
	}
}

func AdamWithName(name string) func(a *Adam) {
	return func(a *Adam) {
		a.name = name
	}
}

type jsonConfigAdam struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (a *Adam) GetKerasLayerConfig() interface{} {
	if a == nil {
		return nil
	}
	return jsonConfigAdam{
		ClassName: "Adam",
		Config: map[string]interface{}{
			"amsgrad":       a.amsgrad,
			"beta_1":        a.beta1,
			"beta_2":        a.beta2,
			"decay":         a.decay,
			"epsilon":       a.epsilon,
			"learning_rate": a.learningRate,
			"name":          a.name,
		},
	}
}

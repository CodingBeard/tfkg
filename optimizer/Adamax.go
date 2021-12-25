package optimizer

type Adamax struct {
	learningRate float64
	beta1        float64
	beta2        float64
	epsilon      float64
	name         string
	decay        float64
}

func NewAdamax() *Adamax {
	return &Adamax{
		learningRate: 0.001,
		beta1:        0.9,
		beta2:        0.999,
		epsilon:      1e-07,
		name:         "Adamax",
	}
}

func AdamaxWithLearningRate(learningRate float64) func(a *Adamax) {
	return func(a *Adamax) {
		a.learningRate = learningRate
	}
}

func AdamaxWithBeta1(beta1 float64) func(a *Adamax) {
	return func(a *Adamax) {
		a.beta1 = beta1
	}
}

func AdamaxWithBeta2(beta2 float64) func(a *Adamax) {
	return func(a *Adamax) {
		a.beta2 = beta2
	}
}

func AdamaxWithEpsilon(epsilon float64) func(a *Adamax) {
	return func(a *Adamax) {
		a.epsilon = epsilon
	}
}

func AdamaxWithName(name string) func(a *Adamax) {
	return func(a *Adamax) {
		a.name = name
	}
}

type jsonConfigAdamax struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (a *Adamax) GetKerasLayerConfig() interface{} {
	if a == nil {
		return nil
	}
	return jsonConfigAdamax{
		ClassName: "Adamax",
		Config: map[string]interface{}{
			"beta_1":        a.beta1,
			"beta_2":        a.beta2,
			"decay":         a.decay,
			"epsilon":       a.epsilon,
			"learning_rate": a.learningRate,
			"name":          a.name,
		},
	}
}

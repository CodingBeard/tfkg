package optimizer

type Nadam struct {
	learningRate float64
	beta1        float64
	beta2        float64
	epsilon      float64
	name         string
	decay        float64
}

func NewNadam() *Nadam {
	return &Nadam{
		learningRate: 0.001,
		beta1:        0.9,
		beta2:        0.999,
		epsilon:      1e-07,
		name:         "Nadam",
	}
}

func NadamWithLearningRate(learningRate float64) func(n *Nadam) {
	return func(n *Nadam) {
		n.learningRate = learningRate
	}
}

func NadamWithBeta1(beta1 float64) func(n *Nadam) {
	return func(n *Nadam) {
		n.beta1 = beta1
	}
}

func NadamWithBeta2(beta2 float64) func(n *Nadam) {
	return func(n *Nadam) {
		n.beta2 = beta2
	}
}

func NadamWithEpsilon(epsilon float64) func(n *Nadam) {
	return func(n *Nadam) {
		n.epsilon = epsilon
	}
}

func NadamWithName(name string) func(n *Nadam) {
	return func(n *Nadam) {
		n.name = name
	}
}

type jsonConfigNadam struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (n *Nadam) GetKerasLayerConfig() interface{} {
	if n == nil {
		return nil
	}
	return jsonConfigNadam{
		ClassName: "Nadam",
		Config: map[string]interface{}{
			"beta_1":        n.beta1,
			"beta_2":        n.beta2,
			"decay":         n.decay,
			"epsilon":       n.epsilon,
			"learning_rate": n.learningRate,
			"name":          n.name,
		},
	}
}

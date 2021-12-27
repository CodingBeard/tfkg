package optimizer

type ONadam struct {
	beta1        float64
	beta2        float64
	decay        float64
	epsilon      float64
	learningRate float64
	name         string
}

func Nadam() *ONadam {
	return &ONadam{
		beta1:        0.9,
		beta2:        0.999,
		decay:        0.004,
		epsilon:      1e-07,
		learningRate: 0.001,
		name:         UniqueName("Nadam"),
	}
}

func (o *ONadam) SetBeta1(beta1 float64) *ONadam {
	o.beta1 = beta1
	return o
}

func (o *ONadam) SetBeta2(beta2 float64) *ONadam {
	o.beta2 = beta2
	return o
}

func (o *ONadam) SetDecay(decay float64) *ONadam {
	o.decay = decay
	return o
}

func (o *ONadam) SetEpsilon(epsilon float64) *ONadam {
	o.epsilon = epsilon
	return o
}

func (o *ONadam) SetLearningRate(learningRate float64) *ONadam {
	o.learningRate = learningRate
	return o
}

func (o *ONadam) SetName(name string) *ONadam {
	o.name = name
	return o
}

type jsonConfigONadam struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (o *ONadam) GetKerasLayerConfig() interface{} {

	return jsonConfigONadam{
		ClassName: "Nadam",
		Name:      o.name,
		Config: map[string]interface{}{
			"beta_1":        o.beta1,
			"beta_2":        o.beta2,
			"decay":         o.decay,
			"epsilon":       o.epsilon,
			"learning_rate": o.learningRate,
			"name":          o.name,
		},
	}
}

func (o *ONadam) GetCustomLayerDefinition() string {
	return ``
}

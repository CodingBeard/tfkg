package optimizer

type OAdamax struct {
	beta1        float64
	beta2        float64
	decay        float64
	epsilon      float64
	learningRate float64
	name         string
}

func Adamax() *OAdamax {
	return &OAdamax{
		beta1:        0.9,
		beta2:        0.999,
		decay:        0,
		epsilon:      1e-07,
		learningRate: 0.001,
		name:         UniqueName("Adamax"),
	}
}

func (o *OAdamax) SetBeta1(beta1 float64) *OAdamax {
	o.beta1 = beta1
	return o
}

func (o *OAdamax) SetBeta2(beta2 float64) *OAdamax {
	o.beta2 = beta2
	return o
}

func (o *OAdamax) SetDecay(decay float64) *OAdamax {
	o.decay = decay
	return o
}

func (o *OAdamax) SetEpsilon(epsilon float64) *OAdamax {
	o.epsilon = epsilon
	return o
}

func (o *OAdamax) SetLearningRate(learningRate float64) *OAdamax {
	o.learningRate = learningRate
	return o
}

func (o *OAdamax) SetName(name string) *OAdamax {
	o.name = name
	return o
}

type jsonConfigOAdamax struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (o *OAdamax) GetKerasLayerConfig() interface{} {

	return jsonConfigOAdamax{
		ClassName: "Adamax",
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

func (o *OAdamax) GetCustomLayerDefinition() string {
	return ``
}

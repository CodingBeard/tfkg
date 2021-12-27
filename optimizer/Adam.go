package optimizer

type OAdam struct {
	amsgrad      bool
	beta1        float64
	beta2        float64
	decay        float64
	epsilon      float64
	learningRate float64
	name         string
}

func Adam() *OAdam {
	return &OAdam{
		amsgrad:      false,
		beta1:        0.9,
		beta2:        0.999,
		decay:        0,
		epsilon:      1e-07,
		learningRate: 0.001,
		name:         UniqueName("Adam"),
	}
}

func (o *OAdam) SetAmsgrad(amsgrad bool) *OAdam {
	o.amsgrad = amsgrad
	return o
}

func (o *OAdam) SetBeta1(beta1 float64) *OAdam {
	o.beta1 = beta1
	return o
}

func (o *OAdam) SetBeta2(beta2 float64) *OAdam {
	o.beta2 = beta2
	return o
}

func (o *OAdam) SetDecay(decay float64) *OAdam {
	o.decay = decay
	return o
}

func (o *OAdam) SetEpsilon(epsilon float64) *OAdam {
	o.epsilon = epsilon
	return o
}

func (o *OAdam) SetLearningRate(learningRate float64) *OAdam {
	o.learningRate = learningRate
	return o
}

func (o *OAdam) SetName(name string) *OAdam {
	o.name = name
	return o
}

type jsonConfigOAdam struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (o *OAdam) GetKerasLayerConfig() interface{} {

	return jsonConfigOAdam{
		ClassName: "Adam",
		Name:      o.name,
		Config: map[string]interface{}{
			"amsgrad":       o.amsgrad,
			"beta_1":        o.beta1,
			"beta_2":        o.beta2,
			"decay":         o.decay,
			"epsilon":       o.epsilon,
			"learning_rate": o.learningRate,
			"name":          o.name,
		},
	}
}

func (o *OAdam) GetCustomLayerDefinition() string {
	return ``
}

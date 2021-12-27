package optimizer

type OAdadelta struct {
	decay        float64
	epsilon      float64
	learningRate float64
	name         string
	rho          float64
}

func Adadelta() *OAdadelta {
	return &OAdadelta{
		decay:        0,
		epsilon:      1e-07,
		learningRate: 0.001,
		name:         UniqueName("Adadelta"),
		rho:          0.95,
	}
}

func (o *OAdadelta) SetDecay(decay float64) *OAdadelta {
	o.decay = decay
	return o
}

func (o *OAdadelta) SetEpsilon(epsilon float64) *OAdadelta {
	o.epsilon = epsilon
	return o
}

func (o *OAdadelta) SetLearningRate(learningRate float64) *OAdadelta {
	o.learningRate = learningRate
	return o
}

func (o *OAdadelta) SetName(name string) *OAdadelta {
	o.name = name
	return o
}

func (o *OAdadelta) SetRho(rho float64) *OAdadelta {
	o.rho = rho
	return o
}

type jsonConfigOAdadelta struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (o *OAdadelta) GetKerasLayerConfig() interface{} {

	return jsonConfigOAdadelta{
		ClassName: "Adadelta",
		Name:      o.name,
		Config: map[string]interface{}{
			"decay":         o.decay,
			"epsilon":       o.epsilon,
			"learning_rate": o.learningRate,
			"name":          o.name,
			"rho":           o.rho,
		},
	}
}

func (o *OAdadelta) GetCustomLayerDefinition() string {
	return ``
}

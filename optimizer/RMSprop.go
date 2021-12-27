package optimizer

type ORMSprop struct {
	centered     bool
	decay        float64
	epsilon      float64
	learningRate float64
	momentum     float64
	name         string
	rho          float64
}

func RMSprop() *ORMSprop {
	return &ORMSprop{
		centered:     false,
		decay:        0,
		epsilon:      1e-07,
		learningRate: 0.001,
		momentum:     0,
		name:         UniqueName("RMSprop"),
		rho:          0.9,
	}
}

func (o *ORMSprop) SetCentered(centered bool) *ORMSprop {
	o.centered = centered
	return o
}

func (o *ORMSprop) SetDecay(decay float64) *ORMSprop {
	o.decay = decay
	return o
}

func (o *ORMSprop) SetEpsilon(epsilon float64) *ORMSprop {
	o.epsilon = epsilon
	return o
}

func (o *ORMSprop) SetLearningRate(learningRate float64) *ORMSprop {
	o.learningRate = learningRate
	return o
}

func (o *ORMSprop) SetMomentum(momentum float64) *ORMSprop {
	o.momentum = momentum
	return o
}

func (o *ORMSprop) SetName(name string) *ORMSprop {
	o.name = name
	return o
}

func (o *ORMSprop) SetRho(rho float64) *ORMSprop {
	o.rho = rho
	return o
}

type jsonConfigORMSprop struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (o *ORMSprop) GetKerasLayerConfig() interface{} {

	return jsonConfigORMSprop{
		ClassName: "RMSprop",
		Name:      o.name,
		Config: map[string]interface{}{
			"centered":      o.centered,
			"decay":         o.decay,
			"epsilon":       o.epsilon,
			"learning_rate": o.learningRate,
			"momentum":      o.momentum,
			"name":          o.name,
			"rho":           o.rho,
		},
	}
}

func (o *ORMSprop) GetCustomLayerDefinition() string {
	return ``
}

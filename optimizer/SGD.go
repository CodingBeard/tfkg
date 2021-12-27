package optimizer

type OSGD struct {
	decay        float64
	learningRate float64
	momentum     float64
	name         string
	nesterov     bool
}

func SGD() *OSGD {
	return &OSGD{
		decay:        0,
		learningRate: 0.01,
		momentum:     0,
		name:         UniqueName("SGD"),
		nesterov:     false,
	}
}

func (o *OSGD) SetDecay(decay float64) *OSGD {
	o.decay = decay
	return o
}

func (o *OSGD) SetLearningRate(learningRate float64) *OSGD {
	o.learningRate = learningRate
	return o
}

func (o *OSGD) SetMomentum(momentum float64) *OSGD {
	o.momentum = momentum
	return o
}

func (o *OSGD) SetName(name string) *OSGD {
	o.name = name
	return o
}

func (o *OSGD) SetNesterov(nesterov bool) *OSGD {
	o.nesterov = nesterov
	return o
}

type jsonConfigOSGD struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (o *OSGD) GetKerasLayerConfig() interface{} {

	return jsonConfigOSGD{
		ClassName: "SGD",
		Name:      o.name,
		Config: map[string]interface{}{
			"decay":         o.decay,
			"learning_rate": o.learningRate,
			"momentum":      o.momentum,
			"name":          o.name,
			"nesterov":      o.nesterov,
		},
	}
}

func (o *OSGD) GetCustomLayerDefinition() string {
	return ``
}

package optimizer

type OAdagrad struct {
	decay                   float64
	epsilon                 float64
	initialAccumulatorValue float64
	learningRate            float64
	name                    string
}

func Adagrad() *OAdagrad {
	return &OAdagrad{
		decay:                   0,
		epsilon:                 1e-07,
		initialAccumulatorValue: 0.1,
		learningRate:            0.001,
		name:                    UniqueName("Adagrad"),
	}
}

func (o *OAdagrad) SetDecay(decay float64) *OAdagrad {
	o.decay = decay
	return o
}

func (o *OAdagrad) SetEpsilon(epsilon float64) *OAdagrad {
	o.epsilon = epsilon
	return o
}

func (o *OAdagrad) SetInitialAccumulatorValue(initialAccumulatorValue float64) *OAdagrad {
	o.initialAccumulatorValue = initialAccumulatorValue
	return o
}

func (o *OAdagrad) SetLearningRate(learningRate float64) *OAdagrad {
	o.learningRate = learningRate
	return o
}

func (o *OAdagrad) SetName(name string) *OAdagrad {
	o.name = name
	return o
}

type jsonConfigOAdagrad struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (o *OAdagrad) GetKerasLayerConfig() interface{} {

	return jsonConfigOAdagrad{
		ClassName: "Adagrad",
		Name:      o.name,
		Config: map[string]interface{}{
			"decay":                     o.decay,
			"epsilon":                   o.epsilon,
			"initial_accumulator_value": o.initialAccumulatorValue,
			"learning_rate":             o.learningRate,
			"name":                      o.name,
		},
	}
}

func (o *OAdagrad) GetCustomLayerDefinition() string {
	return ``
}

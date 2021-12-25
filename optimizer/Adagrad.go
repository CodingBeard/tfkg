package optimizer

type Adagrad struct {
	learningRate            float64
	initialAccumulatorValue float64
	epsilon                 float64
	name                    string
	decay                   float64
}

func NewAdagrad() *Adagrad {
	return &Adagrad{
		learningRate:            0.001,
		initialAccumulatorValue: 0.1,
		epsilon:                 1e-07,
		name:                    "Adagrad",
	}
}

func AdagradWithLearningRate(learningRate float64) func(a *Adagrad) {
	return func(a *Adagrad) {
		a.learningRate = learningRate
	}
}

func AdagradWithInitialAccumulatorValue(initialAccumulatorValue float64) func(a *Adagrad) {
	return func(a *Adagrad) {
		a.initialAccumulatorValue = initialAccumulatorValue
	}
}

func AdagradWithEpsilon(epsilon float64) func(a *Adagrad) {
	return func(a *Adagrad) {
		a.epsilon = epsilon
	}
}

func AdagradWithName(name string) func(a *Adagrad) {
	return func(a *Adagrad) {
		a.name = name
	}
}

type jsonConfigAdagrad struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (a *Adagrad) GetKerasLayerConfig() interface{} {
	if a == nil {
		return nil
	}
	return jsonConfigAdagrad{
		ClassName: "Adagrad",
		Config: map[string]interface{}{
			"decay":                     a.decay,
			"epsilon":                   a.epsilon,
			"initial_accumulator_value": a.initialAccumulatorValue,
			"learning_rate":             a.learningRate,
			"name":                      a.name,
		},
	}
}

package optimizer

type OFtrl struct {
	beta                              float64
	decay                             float64
	initialAccumulatorValue           float64
	l1RegularizationStrength          float64
	l2RegularizationStrength          float64
	l2ShrinkageRegularizationStrength float64
	learningRate                      float64
	learningRatePower                 float64
	name                              string
}

func Ftrl() *OFtrl {
	return &OFtrl{
		beta:                              0,
		decay:                             0,
		initialAccumulatorValue:           0.1,
		l1RegularizationStrength:          0,
		l2RegularizationStrength:          0,
		l2ShrinkageRegularizationStrength: 0,
		learningRate:                      0.001,
		learningRatePower:                 -0.5,
		name:                              UniqueName("Ftrl"),
	}
}

func (o *OFtrl) SetBeta(beta float64) *OFtrl {
	o.beta = beta
	return o
}

func (o *OFtrl) SetDecay(decay float64) *OFtrl {
	o.decay = decay
	return o
}

func (o *OFtrl) SetInitialAccumulatorValue(initialAccumulatorValue float64) *OFtrl {
	o.initialAccumulatorValue = initialAccumulatorValue
	return o
}

func (o *OFtrl) SetL1RegularizationStrength(l1RegularizationStrength float64) *OFtrl {
	o.l1RegularizationStrength = l1RegularizationStrength
	return o
}

func (o *OFtrl) SetL2RegularizationStrength(l2RegularizationStrength float64) *OFtrl {
	o.l2RegularizationStrength = l2RegularizationStrength
	return o
}

func (o *OFtrl) SetL2ShrinkageRegularizationStrength(l2ShrinkageRegularizationStrength float64) *OFtrl {
	o.l2ShrinkageRegularizationStrength = l2ShrinkageRegularizationStrength
	return o
}

func (o *OFtrl) SetLearningRate(learningRate float64) *OFtrl {
	o.learningRate = learningRate
	return o
}

func (o *OFtrl) SetLearningRatePower(learningRatePower float64) *OFtrl {
	o.learningRatePower = learningRatePower
	return o
}

func (o *OFtrl) SetName(name string) *OFtrl {
	o.name = name
	return o
}

type jsonConfigOFtrl struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (o *OFtrl) GetKerasLayerConfig() interface{} {

	return jsonConfigOFtrl{
		ClassName: "Ftrl",
		Name:      o.name,
		Config: map[string]interface{}{
			"beta":                                 o.beta,
			"decay":                                o.decay,
			"initial_accumulator_value":            o.initialAccumulatorValue,
			"l1_regularization_strength":           o.l1RegularizationStrength,
			"l2_regularization_strength":           o.l2RegularizationStrength,
			"l2_shrinkage_regularization_strength": o.l2ShrinkageRegularizationStrength,
			"learning_rate":                        o.learningRate,
			"learning_rate_power":                  o.learningRatePower,
			"name":                                 o.name,
		},
	}
}

func (o *OFtrl) GetCustomLayerDefinition() string {
	return ``
}

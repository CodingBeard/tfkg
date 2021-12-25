package optimizer

type Ftrl struct {
	learningRate                      float64
	learningRatePower                 float64
	initialAccumulatorValue           float64
	l1RegularizationStrength          float64
	l2RegularizationStrength          float64
	name                              string
	l2ShrinkageRegularizationStrength float64
	beta                              float64
	decay                             float64
}

func NewFtrl() *Ftrl {
	return &Ftrl{
		learningRate:                      0.001,
		learningRatePower:                 -0.5,
		initialAccumulatorValue:           0.1,
		l1RegularizationStrength:          0,
		l2RegularizationStrength:          0,
		name:                              "Ftrl",
		l2ShrinkageRegularizationStrength: 0,
		beta:                              0,
	}
}

func FtrlWithLearningRate(learningRate float64) func(f *Ftrl) {
	return func(f *Ftrl) {
		f.learningRate = learningRate
	}
}

func FtrlWithLearningRatePower(learningRatePower float64) func(f *Ftrl) {
	return func(f *Ftrl) {
		f.learningRatePower = learningRatePower
	}
}

func FtrlWithInitialAccumulatorValue(initialAccumulatorValue float64) func(f *Ftrl) {
	return func(f *Ftrl) {
		f.initialAccumulatorValue = initialAccumulatorValue
	}
}

func FtrlWithL1RegularizationStrength(l1RegularizationStrength float64) func(f *Ftrl) {
	return func(f *Ftrl) {
		f.l1RegularizationStrength = l1RegularizationStrength
	}
}

func FtrlWithL2RegularizationStrength(l2RegularizationStrength float64) func(f *Ftrl) {
	return func(f *Ftrl) {
		f.l2RegularizationStrength = l2RegularizationStrength
	}
}

func FtrlWithName(name string) func(f *Ftrl) {
	return func(f *Ftrl) {
		f.name = name
	}
}

func FtrlWithL2ShrinkageRegularizationStrength(l2ShrinkageRegularizationStrength float64) func(f *Ftrl) {
	return func(f *Ftrl) {
		f.l2ShrinkageRegularizationStrength = l2ShrinkageRegularizationStrength
	}
}

func FtrlWithBeta(beta float64) func(f *Ftrl) {
	return func(f *Ftrl) {
		f.beta = beta
	}
}

type jsonConfigFtrl struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (f *Ftrl) GetKerasLayerConfig() interface{} {
	if f == nil {
		return nil
	}
	return jsonConfigFtrl{
		ClassName: "Ftrl",
		Config: map[string]interface{}{
			"beta":                                 f.beta,
			"decay":                                f.decay,
			"initial_accumulator_value":            f.initialAccumulatorValue,
			"l1_regularization_strength":           f.l1RegularizationStrength,
			"l2_regularization_strength":           f.l2RegularizationStrength,
			"l2_shrinkage_regularization_strength": f.l2ShrinkageRegularizationStrength,
			"learning_rate":                        f.learningRate,
			"learning_rate_power":                  f.learningRatePower,
			"name":                                 f.name,
		},
	}
}

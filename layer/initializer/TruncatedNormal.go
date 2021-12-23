package initializer

type TruncatedNormal struct {
	mean   float64
	stddev float64
	seed   interface{}
}

func NewTruncatedNormal() *TruncatedNormal {
	return &TruncatedNormal{
		mean:   0,
		stddev: 0.05,
		seed:   nil,
	}
}

func TruncatedNormalWithMean(mean float64) func(t *TruncatedNormal) {
	return func(t *TruncatedNormal) {
		t.mean = mean
	}
}

func TruncatedNormalWithStddev(stddev float64) func(t *TruncatedNormal) {
	return func(t *TruncatedNormal) {
		t.stddev = stddev
	}
}

func TruncatedNormalWithSeed(seed interface{}) func(t *TruncatedNormal) {
	return func(t *TruncatedNormal) {
		t.seed = seed
	}
}

type jsonConfigTruncatedNormal struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (t *TruncatedNormal) GetKerasLayerConfig() interface{} {
	if t == nil {
		return nil
	}
	return jsonConfigTruncatedNormal{
		ClassName: "TruncatedNormal",
		Config: map[string]interface{}{
			"mean":   t.mean,
			"seed":   t.seed,
			"stddev": t.stddev,
		},
	}
}

package initializer

type RandomNormal struct {
	mean   float64
	stddev float64
	seed   interface{}
}

func NewRandomNormal() *RandomNormal {
	return &RandomNormal{
		mean:   0,
		stddev: 0.05,
		seed:   nil,
	}
}

func RandomNormalWithMean(mean float64) func(r *RandomNormal) {
	return func(r *RandomNormal) {
		r.mean = mean
	}
}

func RandomNormalWithStddev(stddev float64) func(r *RandomNormal) {
	return func(r *RandomNormal) {
		r.stddev = stddev
	}
}

func RandomNormalWithSeed(seed interface{}) func(r *RandomNormal) {
	return func(r *RandomNormal) {
		r.seed = seed
	}
}

type jsonConfigRandomNormal struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (r *RandomNormal) GetKerasLayerConfig() interface{} {
	if r == nil {
		return nil
	}
	return jsonConfigRandomNormal{
		ClassName: "RandomNormal",
		Config: map[string]interface{}{
			"mean":   r.mean,
			"stddev": r.stddev,
			"seed":   r.seed,
		},
	}
}

package initializer

type RandomUniform struct {
	minval float64
	maxval float64
	seed   interface{}
}

func NewRandomUniform() *RandomUniform {
	return &RandomUniform{
		minval: -0.05,
		maxval: 0.05,
		seed:   nil,
	}
}

func RandomUniformWithMinval(minval float64) func(r *RandomUniform) {
	return func(r *RandomUniform) {
		r.minval = minval
	}
}

func RandomUniformWithMaxval(maxval float64) func(r *RandomUniform) {
	return func(r *RandomUniform) {
		r.maxval = maxval
	}
}

func RandomUniformWithSeed(seed interface{}) func(r *RandomUniform) {
	return func(r *RandomUniform) {
		r.seed = seed
	}
}

type jsonConfigRandomUniform struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (r *RandomUniform) GetKerasLayerConfig() interface{} {
	if r == nil {
		return nil
	}
	return jsonConfigRandomUniform{
		ClassName: "RandomUniform",
		Config: map[string]interface{}{
			"maxval": r.maxval,
			"minval": r.minval,
			"seed":   r.seed,
		},
	}
}

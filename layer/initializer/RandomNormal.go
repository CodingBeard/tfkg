package initializer

type IRandomNormal struct {
	mean   float64
	name   string
	seed   interface{}
	stddev float64
}

func RandomNormal() *IRandomNormal {
	return &IRandomNormal{
		mean:   0,
		seed:   nil,
		stddev: 0.05,
	}
}

func (i *IRandomNormal) SetMean(mean float64) *IRandomNormal {
	i.mean = mean
	return i
}

func (i *IRandomNormal) SetName(name string) *IRandomNormal {
	i.name = name
	return i
}

func (i *IRandomNormal) SetSeed(seed interface{}) *IRandomNormal {
	i.seed = seed
	return i
}

func (i *IRandomNormal) SetStddev(stddev float64) *IRandomNormal {
	i.stddev = stddev
	return i
}

type jsonConfigIRandomNormal struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (i *IRandomNormal) GetKerasLayerConfig() interface{} {

	return jsonConfigIRandomNormal{
		ClassName: "RandomNormal",
		Name:      i.name,
		Config: map[string]interface{}{
			"mean":   i.mean,
			"seed":   i.seed,
			"stddev": i.stddev,
		},
	}
}

func (i *IRandomNormal) GetCustomLayerDefinition() string {
	return ``
}

package initializer

type ITruncatedNormal struct {
	mean   float64
	name   string
	seed   interface{}
	stddev float64
}

func TruncatedNormal() *ITruncatedNormal {
	return &ITruncatedNormal{
		mean:   0,
		seed:   nil,
		stddev: 0.05,
	}
}

func (i *ITruncatedNormal) SetMean(mean float64) *ITruncatedNormal {
	i.mean = mean
	return i
}

func (i *ITruncatedNormal) SetName(name string) *ITruncatedNormal {
	i.name = name
	return i
}

func (i *ITruncatedNormal) SetSeed(seed interface{}) *ITruncatedNormal {
	i.seed = seed
	return i
}

func (i *ITruncatedNormal) SetStddev(stddev float64) *ITruncatedNormal {
	i.stddev = stddev
	return i
}

type jsonConfigITruncatedNormal struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (i *ITruncatedNormal) GetKerasLayerConfig() interface{} {

	return jsonConfigITruncatedNormal{
		ClassName: "TruncatedNormal",
		Name:      i.name,
		Config: map[string]interface{}{
			"mean":   i.mean,
			"seed":   i.seed,
			"stddev": i.stddev,
		},
	}
}

func (i *ITruncatedNormal) GetCustomLayerDefinition() string {
	return ``
}

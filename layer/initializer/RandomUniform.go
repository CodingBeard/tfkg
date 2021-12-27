package initializer

type IRandomUniform struct {
	maxval float64
	minval float64
	name   string
	seed   interface{}
}

func RandomUniform() *IRandomUniform {
	return &IRandomUniform{
		maxval: 0.05,
		minval: -0.05,
		seed:   nil,
	}
}

func (i *IRandomUniform) SetMaxval(maxval float64) *IRandomUniform {
	i.maxval = maxval
	return i
}

func (i *IRandomUniform) SetMinval(minval float64) *IRandomUniform {
	i.minval = minval
	return i
}

func (i *IRandomUniform) SetName(name string) *IRandomUniform {
	i.name = name
	return i
}

func (i *IRandomUniform) SetSeed(seed interface{}) *IRandomUniform {
	i.seed = seed
	return i
}

type jsonConfigIRandomUniform struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (i *IRandomUniform) GetKerasLayerConfig() interface{} {

	return jsonConfigIRandomUniform{
		ClassName: "RandomUniform",
		Name:      i.name,
		Config: map[string]interface{}{
			"maxval": i.maxval,
			"minval": i.minval,
			"seed":   i.seed,
		},
	}
}

func (i *IRandomUniform) GetCustomLayerDefinition() string {
	return ``
}

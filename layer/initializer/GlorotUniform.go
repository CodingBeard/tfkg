package initializer

type IGlorotUniform struct {
	name string
	seed interface{}
}

func GlorotUniform() *IGlorotUniform {
	return &IGlorotUniform{
		seed: nil,
	}
}

func (i *IGlorotUniform) SetName(name string) *IGlorotUniform {
	i.name = name
	return i
}

func (i *IGlorotUniform) SetSeed(seed interface{}) *IGlorotUniform {
	i.seed = seed
	return i
}

type jsonConfigIGlorotUniform struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (i *IGlorotUniform) GetKerasLayerConfig() interface{} {

	return jsonConfigIGlorotUniform{
		ClassName: "GlorotUniform",
		Name:      i.name,
		Config: map[string]interface{}{
			"seed": i.seed,
		},
	}
}

func (i *IGlorotUniform) GetCustomLayerDefinition() string {
	return ``
}

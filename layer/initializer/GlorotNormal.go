package initializer

type IGlorotNormal struct {
	name string
	seed interface{}
}

func GlorotNormal() *IGlorotNormal {
	return &IGlorotNormal{
		seed: nil,
	}
}

func (i *IGlorotNormal) SetName(name string) *IGlorotNormal {
	i.name = name
	return i
}

func (i *IGlorotNormal) SetSeed(seed interface{}) *IGlorotNormal {
	i.seed = seed
	return i
}

type jsonConfigIGlorotNormal struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (i *IGlorotNormal) GetKerasLayerConfig() interface{} {

	return jsonConfigIGlorotNormal{
		ClassName: "GlorotNormal",
		Name:      i.name,
		Config: map[string]interface{}{
			"seed": i.seed,
		},
	}
}

func (i *IGlorotNormal) GetCustomLayerDefinition() string {
	return ``
}

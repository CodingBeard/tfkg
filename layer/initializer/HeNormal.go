package initializer

type IHeNormal struct {
	name string
	seed interface{}
}

func HeNormal() *IHeNormal {
	return &IHeNormal{
		seed: nil,
	}
}

func (i *IHeNormal) SetName(name string) *IHeNormal {
	i.name = name
	return i
}

func (i *IHeNormal) SetSeed(seed interface{}) *IHeNormal {
	i.seed = seed
	return i
}

type jsonConfigIHeNormal struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (i *IHeNormal) GetKerasLayerConfig() interface{} {

	return jsonConfigIHeNormal{
		ClassName: "HeNormal",
		Name:      i.name,
		Config: map[string]interface{}{
			"seed": i.seed,
		},
	}
}

func (i *IHeNormal) GetCustomLayerDefinition() string {
	return ``
}

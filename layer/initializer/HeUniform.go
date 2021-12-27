package initializer

type IHeUniform struct {
	name string
	seed interface{}
}

func HeUniform() *IHeUniform {
	return &IHeUniform{
		seed: nil,
	}
}

func (i *IHeUniform) SetName(name string) *IHeUniform {
	i.name = name
	return i
}

func (i *IHeUniform) SetSeed(seed interface{}) *IHeUniform {
	i.seed = seed
	return i
}

type jsonConfigIHeUniform struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (i *IHeUniform) GetKerasLayerConfig() interface{} {

	return jsonConfigIHeUniform{
		ClassName: "HeUniform",
		Name:      i.name,
		Config: map[string]interface{}{
			"seed": i.seed,
		},
	}
}

func (i *IHeUniform) GetCustomLayerDefinition() string {
	return ``
}

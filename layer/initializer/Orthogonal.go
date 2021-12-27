package initializer

type IOrthogonal struct {
	gain float64
	name string
	seed interface{}
}

func Orthogonal() *IOrthogonal {
	return &IOrthogonal{
		gain: 1,
		seed: nil,
	}
}

func (i *IOrthogonal) SetGain(gain float64) *IOrthogonal {
	i.gain = gain
	return i
}

func (i *IOrthogonal) SetName(name string) *IOrthogonal {
	i.name = name
	return i
}

func (i *IOrthogonal) SetSeed(seed interface{}) *IOrthogonal {
	i.seed = seed
	return i
}

type jsonConfigIOrthogonal struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (i *IOrthogonal) GetKerasLayerConfig() interface{} {

	return jsonConfigIOrthogonal{
		ClassName: "Orthogonal",
		Name:      i.name,
		Config: map[string]interface{}{
			"gain": i.gain,
			"seed": i.seed,
		},
	}
}

func (i *IOrthogonal) GetCustomLayerDefinition() string {
	return ``
}

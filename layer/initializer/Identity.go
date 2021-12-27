package initializer

type IIdentity struct {
	gain float64
	name string
}

func Identity() *IIdentity {
	return &IIdentity{
		gain: 1,
	}
}

func (i *IIdentity) SetGain(gain float64) *IIdentity {
	i.gain = gain
	return i
}

func (i *IIdentity) SetName(name string) *IIdentity {
	i.name = name
	return i
}

type jsonConfigIIdentity struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (i *IIdentity) GetKerasLayerConfig() interface{} {

	return jsonConfigIIdentity{
		ClassName: "Identity",
		Name:      i.name,
		Config: map[string]interface{}{
			"gain": i.gain,
		},
	}
}

func (i *IIdentity) GetCustomLayerDefinition() string {
	return ``
}

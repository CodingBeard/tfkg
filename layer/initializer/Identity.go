package initializer

type Identity struct {
	gain float64
}

func NewIdentity() *Identity {
	return &Identity{
		gain: 1,
	}
}

func IdentityWithGain(gain float64) func(i *Identity) {
	return func(i *Identity) {
		i.gain = gain
	}
}

type jsonConfigIdentity struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (i *Identity) GetKerasLayerConfig() interface{} {
	if i == nil {
		return nil
	}
	return jsonConfigIdentity{
		ClassName: "Identity",
		Config: map[string]interface{}{
			"gain": i.gain,
		},
	}
}

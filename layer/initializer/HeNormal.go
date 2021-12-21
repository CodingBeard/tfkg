package initializer

type HeNormal struct {
	seed interface{}
}

func NewHeNormal() *HeNormal {
	return &HeNormal{
		seed: nil,
	}
}

func HeNormalWithSeed(seed interface{}) func(h *HeNormal) {
	return func(h *HeNormal) {
		h.seed = seed
	}
}

type jsonConfigHeNormal struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (h *HeNormal) GetKerasLayerConfig() interface{} {
	if h == nil {
		return nil
	}
	return jsonConfigHeNormal{
		ClassName: "HeNormal",
		Config: map[string]interface{}{
			"seed": h.seed,
		},
	}
}

package initializer

type HeUniform struct {
	seed interface{}
}

func NewHeUniform() *HeUniform {
	return &HeUniform{
		seed: nil,
	}
}

func HeUniformWithSeed(seed interface{}) func(h *HeUniform) {
	return func(h *HeUniform) {
		h.seed = seed
	}
}

type jsonConfigHeUniform struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (h *HeUniform) GetKerasLayerConfig() interface{} {
	if h == nil {
		return nil
	}
	return jsonConfigHeUniform{
		ClassName: "HeUniform",
		Config: map[string]interface{}{
			"seed": h.seed,
		},
	}
}

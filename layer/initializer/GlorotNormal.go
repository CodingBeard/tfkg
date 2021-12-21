package initializer

type GlorotNormal struct {
	seed interface{}
}

func NewGlorotNormal() *GlorotNormal {
	return &GlorotNormal{
		seed: nil,
	}
}

func GlorotNormalWithSeed(seed interface{}) func(g *GlorotNormal) {
	return func(g *GlorotNormal) {
		g.seed = seed
	}
}

type jsonConfigGlorotNormal struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (g *GlorotNormal) GetKerasLayerConfig() interface{} {
	if g == nil {
		return nil
	}
	return jsonConfigGlorotNormal{
		ClassName: "GlorotNormal",
		Config: map[string]interface{}{
			"seed": g.seed,
		},
	}
}

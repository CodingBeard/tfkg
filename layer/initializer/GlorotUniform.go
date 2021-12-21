package initializer

type GlorotUniform struct {
	seed interface{}
}

func NewGlorotUniform() *GlorotUniform {
	return &GlorotUniform{
		seed: nil,
	}
}

func GlorotUniformWithSeed(seed interface{}) func(g *GlorotUniform) {
	return func(g *GlorotUniform) {
		g.seed = seed
	}
}

type jsonConfigGlorotUniform struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (g *GlorotUniform) GetKerasLayerConfig() interface{} {
	if g == nil {
		return nil
	}
	return jsonConfigGlorotUniform{
		ClassName: "GlorotUniform",
		Config: map[string]interface{}{
			"seed": g.seed,
		},
	}
}

package initializer

type IVarianceScaling struct {
	distribution string
	mode         string
	name         string
	scale        float64
	seed         interface{}
}

func VarianceScaling() *IVarianceScaling {
	return &IVarianceScaling{
		distribution: "truncated_normal",
		mode:         "fan_in",
		scale:        1,
		seed:         nil,
	}
}

func (i *IVarianceScaling) SetDistribution(distribution string) *IVarianceScaling {
	i.distribution = distribution
	return i
}

func (i *IVarianceScaling) SetMode(mode string) *IVarianceScaling {
	i.mode = mode
	return i
}

func (i *IVarianceScaling) SetName(name string) *IVarianceScaling {
	i.name = name
	return i
}

func (i *IVarianceScaling) SetScale(scale float64) *IVarianceScaling {
	i.scale = scale
	return i
}

func (i *IVarianceScaling) SetSeed(seed interface{}) *IVarianceScaling {
	i.seed = seed
	return i
}

type jsonConfigIVarianceScaling struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (i *IVarianceScaling) GetKerasLayerConfig() interface{} {

	return jsonConfigIVarianceScaling{
		ClassName: "VarianceScaling",
		Name:      i.name,
		Config: map[string]interface{}{
			"distribution": i.distribution,
			"mode":         i.mode,
			"scale":        i.scale,
			"seed":         i.seed,
		},
	}
}

func (i *IVarianceScaling) GetCustomLayerDefinition() string {
	return ``
}

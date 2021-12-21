package initializer

type VarianceScaling struct {
	scale        float64
	mode         string
	distribution string
	seed         interface{}
}

func NewVarianceScaling() *VarianceScaling {
	return &VarianceScaling{
		scale:        1,
		mode:         "fan_in",
		distribution: "truncated_normal",
		seed:         nil,
	}
}

func VarianceScalingWithScale(scale float64) func(v *VarianceScaling) {
	return func(v *VarianceScaling) {
		v.scale = scale
	}
}

func VarianceScalingWithMode(mode string) func(v *VarianceScaling) {
	return func(v *VarianceScaling) {
		v.mode = mode
	}
}

func VarianceScalingWithDistribution(distribution string) func(v *VarianceScaling) {
	return func(v *VarianceScaling) {
		v.distribution = distribution
	}
}

func VarianceScalingWithSeed(seed interface{}) func(v *VarianceScaling) {
	return func(v *VarianceScaling) {
		v.seed = seed
	}
}

type jsonConfigVarianceScaling struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (v *VarianceScaling) GetKerasLayerConfig() interface{} {
	if v == nil {
		return nil
	}
	return jsonConfigVarianceScaling{
		ClassName: "VarianceScaling",
		Config: map[string]interface{}{
			"mode":         v.mode,
			"distribution": v.distribution,
			"seed":         v.seed,
			"scale":        v.scale,
		},
	}
}

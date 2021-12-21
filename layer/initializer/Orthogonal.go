package initializer

type Orthogonal struct {
	gain float64
	seed interface{}
}

func NewOrthogonal() *Orthogonal {
	return &Orthogonal{
		gain: 1,
		seed: nil,
	}
}

func OrthogonalWithGain(gain float64) func(o *Orthogonal) {
	return func(o *Orthogonal) {
		o.gain = gain
	}
}

func OrthogonalWithSeed(seed interface{}) func(o *Orthogonal) {
	return func(o *Orthogonal) {
		o.seed = seed
	}
}

type jsonConfigOrthogonal struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (o *Orthogonal) GetKerasLayerConfig() interface{} {
	if o == nil {
		return nil
	}
	return jsonConfigOrthogonal{
		ClassName: "Orthogonal",
		Config: map[string]interface{}{
			"gain": o.gain,
			"seed": o.seed,
		},
	}
}

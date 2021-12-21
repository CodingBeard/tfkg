package constraint

type MaxNorm struct {
	maxValue float64
	axis     float64
}

func NewMaxNorm() *MaxNorm {
	return &MaxNorm{
		maxValue: 2,
		axis:     0,
	}
}

func MaxNormWithMaxValue(maxValue float64) func(m *MaxNorm) {
	return func(m *MaxNorm) {
		m.maxValue = maxValue
	}
}

func MaxNormWithAxis(axis float64) func(m *MaxNorm) {
	return func(m *MaxNorm) {
		m.axis = axis
	}
}

type jsonConfigMaxNorm struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (m *MaxNorm) GetKerasLayerConfig() interface{} {
	if m == nil {
		return nil
	}
	return jsonConfigMaxNorm{
		ClassName: "MaxNorm",
		Config: map[string]interface{}{
			"max_value": m.maxValue,
			"axis":      m.axis,
		},
	}
}

package constraint

type MinMaxNorm struct {
	minValue float64
	maxValue float64
	rate     float64
	axis     float64
}

func NewMinMaxNorm() *MinMaxNorm {
	return &MinMaxNorm{
		minValue: 0,
		maxValue: 1,
		rate:     1,
		axis:     0,
	}
}

func MinMaxNormWithMinValue(minValue float64) func(m *MinMaxNorm) {
	return func(m *MinMaxNorm) {
		m.minValue = minValue
	}
}

func MinMaxNormWithMaxValue(maxValue float64) func(m *MinMaxNorm) {
	return func(m *MinMaxNorm) {
		m.maxValue = maxValue
	}
}

func MinMaxNormWithRate(rate float64) func(m *MinMaxNorm) {
	return func(m *MinMaxNorm) {
		m.rate = rate
	}
}

func MinMaxNormWithAxis(axis float64) func(m *MinMaxNorm) {
	return func(m *MinMaxNorm) {
		m.axis = axis
	}
}

type jsonConfigMinMaxNorm struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (m *MinMaxNorm) GetKerasLayerConfig() interface{} {
	if m == nil {
		return nil
	}
	return jsonConfigMinMaxNorm{
		ClassName: "MinMaxNorm",
		Config: map[string]interface{}{
			"min_value": m.minValue,
			"max_value": m.maxValue,
			"rate":      m.rate,
			"axis":      m.axis,
		},
	}
}

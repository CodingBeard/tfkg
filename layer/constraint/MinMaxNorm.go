package constraint

type CMinMaxNorm struct {
	axis     float64
	maxValue float64
	minValue float64
	name     string
	rate     float64
}

func MinMaxNorm() *CMinMaxNorm {
	return &CMinMaxNorm{
		axis:     0,
		maxValue: 1,
		minValue: 0,
		rate:     1,
	}
}

func (c *CMinMaxNorm) SetAxis(axis float64) *CMinMaxNorm {
	c.axis = axis
	return c
}

func (c *CMinMaxNorm) SetMaxValue(maxValue float64) *CMinMaxNorm {
	c.maxValue = maxValue
	return c
}

func (c *CMinMaxNorm) SetMinValue(minValue float64) *CMinMaxNorm {
	c.minValue = minValue
	return c
}

func (c *CMinMaxNorm) SetName(name string) *CMinMaxNorm {
	c.name = name
	return c
}

func (c *CMinMaxNorm) SetRate(rate float64) *CMinMaxNorm {
	c.rate = rate
	return c
}

type jsonConfigCMinMaxNorm struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (c *CMinMaxNorm) GetKerasLayerConfig() interface{} {

	return jsonConfigCMinMaxNorm{
		ClassName: "MinMaxNorm",
		Name:      c.name,
		Config: map[string]interface{}{
			"axis":      c.axis,
			"max_value": c.maxValue,
			"min_value": c.minValue,
			"rate":      c.rate,
		},
	}
}

func (c *CMinMaxNorm) GetCustomLayerDefinition() string {
	return ``
}

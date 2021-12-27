package constraint

type CMaxNorm struct {
	axis     float64
	maxValue float64
	name     string
}

func MaxNorm() *CMaxNorm {
	return &CMaxNorm{
		axis:     0,
		maxValue: 2,
	}
}

func (c *CMaxNorm) SetAxis(axis float64) *CMaxNorm {
	c.axis = axis
	return c
}

func (c *CMaxNorm) SetMaxValue(maxValue float64) *CMaxNorm {
	c.maxValue = maxValue
	return c
}

func (c *CMaxNorm) SetName(name string) *CMaxNorm {
	c.name = name
	return c
}

type jsonConfigCMaxNorm struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (c *CMaxNorm) GetKerasLayerConfig() interface{} {

	return jsonConfigCMaxNorm{
		ClassName: "MaxNorm",
		Name:      c.name,
		Config: map[string]interface{}{
			"axis":      c.axis,
			"max_value": c.maxValue,
		},
	}
}

func (c *CMaxNorm) GetCustomLayerDefinition() string {
	return ``
}

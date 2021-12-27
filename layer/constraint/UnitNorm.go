package constraint

type CUnitNorm struct {
	axis float64
	name string
}

func UnitNorm() *CUnitNorm {
	return &CUnitNorm{
		axis: 0,
	}
}

func (c *CUnitNorm) SetAxis(axis float64) *CUnitNorm {
	c.axis = axis
	return c
}

func (c *CUnitNorm) SetName(name string) *CUnitNorm {
	c.name = name
	return c
}

type jsonConfigCUnitNorm struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (c *CUnitNorm) GetKerasLayerConfig() interface{} {

	return jsonConfigCUnitNorm{
		ClassName: "UnitNorm",
		Name:      c.name,
		Config: map[string]interface{}{
			"axis": c.axis,
		},
	}
}

func (c *CUnitNorm) GetCustomLayerDefinition() string {
	return ``
}

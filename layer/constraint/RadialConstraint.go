package constraint

type CRadialConstraint struct {
	name string
}

func RadialConstraint() *CRadialConstraint {
	return &CRadialConstraint{}
}

func (c *CRadialConstraint) SetName(name string) *CRadialConstraint {
	c.name = name
	return c
}

type jsonConfigCRadialConstraint struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (c *CRadialConstraint) GetKerasLayerConfig() interface{} {

	return jsonConfigCRadialConstraint{
		ClassName: "RadialConstraint",
		Name:      c.name,
		Config:    map[string]interface{}{},
	}
}

func (c *CRadialConstraint) GetCustomLayerDefinition() string {
	return ``
}

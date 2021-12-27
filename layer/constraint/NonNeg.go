package constraint

type CNonNeg struct {
	name string
}

func NonNeg() *CNonNeg {
	return &CNonNeg{}
}

func (c *CNonNeg) SetName(name string) *CNonNeg {
	c.name = name
	return c
}

type jsonConfigCNonNeg struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (c *CNonNeg) GetKerasLayerConfig() interface{} {

	return jsonConfigCNonNeg{
		ClassName: "NonNeg",
		Name:      c.name,
		Config:    map[string]interface{}{},
	}
}

func (c *CNonNeg) GetCustomLayerDefinition() string {
	return ``
}

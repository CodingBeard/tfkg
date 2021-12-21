package initializer

type Constant struct {
	value float64
}

func NewConstant() *Constant {
	return &Constant{
		value: 0,
	}
}

func ConstantWithValue(value float64) func(c *Constant) {
	return func(c *Constant) {
		c.value = value
	}
}

type jsonConfigConstant struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (c *Constant) GetKerasLayerConfig() interface{} {
	if c == nil {
		return nil
	}
	return jsonConfigConstant{
		ClassName: "Constant",
		Config: map[string]interface{}{
			"value": c.value,
		},
	}
}

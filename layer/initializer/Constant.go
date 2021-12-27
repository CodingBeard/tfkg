package initializer

type IConstant struct {
	name  string
	value float64
}

func Constant() *IConstant {
	return &IConstant{
		value: 0,
	}
}

func (i *IConstant) SetName(name string) *IConstant {
	i.name = name
	return i
}

func (i *IConstant) SetValue(value float64) *IConstant {
	i.value = value
	return i
}

type jsonConfigIConstant struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (i *IConstant) GetKerasLayerConfig() interface{} {

	return jsonConfigIConstant{
		ClassName: "Constant",
		Name:      i.name,
		Config: map[string]interface{}{
			"value": i.value,
		},
	}
}

func (i *IConstant) GetCustomLayerDefinition() string {
	return ``
}

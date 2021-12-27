package initializer

type IOnes struct {
	name string
}

func Ones() *IOnes {
	return &IOnes{}
}

func (i *IOnes) SetName(name string) *IOnes {
	i.name = name
	return i
}

type jsonConfigIOnes struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (i *IOnes) GetKerasLayerConfig() interface{} {

	return jsonConfigIOnes{
		ClassName: "Ones",
		Name:      i.name,
		Config:    map[string]interface{}{},
	}
}

func (i *IOnes) GetCustomLayerDefinition() string {
	return ``
}

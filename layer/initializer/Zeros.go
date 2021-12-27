package initializer

type IZeros struct {
	name string
}

func Zeros() *IZeros {
	return &IZeros{}
}

func (i *IZeros) SetName(name string) *IZeros {
	i.name = name
	return i
}

type jsonConfigIZeros struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (i *IZeros) GetKerasLayerConfig() interface{} {

	return jsonConfigIZeros{
		ClassName: "Zeros",
		Name:      i.name,
		Config:    map[string]interface{}{},
	}
}

func (i *IZeros) GetCustomLayerDefinition() string {
	return ``
}

package initializer

type Zeros struct {
}

func NewZeros() *Zeros {
	return &Zeros{}
}

type jsonConfigZeros struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (z *Zeros) GetKerasLayerConfig() interface{} {
	if z == nil {
		return nil
	}
	return jsonConfigZeros{
		ClassName: "Zeros",
		Config:    map[string]interface{}{},
	}
}

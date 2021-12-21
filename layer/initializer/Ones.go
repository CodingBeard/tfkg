package initializer

type Ones struct {
}

func NewOnes() *Ones {
	return &Ones{}
}

type jsonConfigOnes struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (o *Ones) GetKerasLayerConfig() interface{} {
	if o == nil {
		return nil
	}
	return jsonConfigOnes{
		ClassName: "Ones",
		Config:    map[string]interface{}{},
	}
}

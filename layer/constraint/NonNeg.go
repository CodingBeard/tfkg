package constraint

type NonNeg struct {
}

func NewNonNeg() *NonNeg {
	return &NonNeg{}
}

type jsonConfigNonNeg struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (n *NonNeg) GetKerasLayerConfig() interface{} {
	if n == nil {
		return nil
	}
	return jsonConfigNonNeg{
		ClassName: "NonNeg",
		Config:    map[string]interface{}{},
	}
}

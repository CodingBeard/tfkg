package constraint

type RadialConstraint struct {
}

func NewRadialConstraint() *RadialConstraint {
	return &RadialConstraint{}
}

type jsonConfigRadialConstraint struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (r *RadialConstraint) GetKerasLayerConfig() interface{} {
	if r == nil {
		return nil
	}
	return jsonConfigRadialConstraint{
		ClassName: "RadialConstraint",
		Config:    map[string]interface{}{},
	}
}

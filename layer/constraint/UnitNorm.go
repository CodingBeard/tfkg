package constraint

type UnitNorm struct {
	axis float64
}

func NewUnitNorm() *UnitNorm {
	return &UnitNorm{
		axis: 0,
	}
}

func UnitNormWithAxis(axis float64) func(u *UnitNorm) {
	return func(u *UnitNorm) {
		u.axis = axis
	}
}

type jsonConfigUnitNorm struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (u *UnitNorm) GetKerasLayerConfig() interface{} {
	if u == nil {
		return nil
	}
	return jsonConfigUnitNorm{
		ClassName: "UnitNorm",
		Config: map[string]interface{}{
			"axis": u.axis,
		},
	}
}

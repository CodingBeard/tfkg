package regularizer

type RL1 struct {
	l1   float64
	name string
}

func L1() *RL1 {
	return &RL1{
		l1: 0.01,
	}
}

func (r *RL1) SetL1(l1 float64) *RL1 {
	r.l1 = l1
	return r
}

func (r *RL1) SetName(name string) *RL1 {
	r.name = name
	return r
}

type jsonConfigRL1 struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (r *RL1) GetKerasLayerConfig() interface{} {

	return jsonConfigRL1{
		ClassName: "L1",
		Name:      r.name,
		Config: map[string]interface{}{
			"l1": r.l1,
		},
	}
}

func (r *RL1) GetCustomLayerDefinition() string {
	return ``
}

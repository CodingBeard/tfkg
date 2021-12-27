package regularizer

type RL2 struct {
	l2   float64
	name string
}

func L2() *RL2 {
	return &RL2{
		l2: 0.01,
	}
}

func (r *RL2) SetL2(l2 float64) *RL2 {
	r.l2 = l2
	return r
}

func (r *RL2) SetName(name string) *RL2 {
	r.name = name
	return r
}

type jsonConfigRL2 struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (r *RL2) GetKerasLayerConfig() interface{} {

	return jsonConfigRL2{
		ClassName: "L2",
		Name:      r.name,
		Config: map[string]interface{}{
			"l2": r.l2,
		},
	}
}

func (r *RL2) GetCustomLayerDefinition() string {
	return ``
}

package regularizer

type L2 struct {
	l2 float64
}

func NewL2() *L2 {
	return &L2{
		l2: 0.01,
	}
}

func L2WithL2(l2 float64) func(l *L2) {
	return func(l *L2) {
		l.l2 = l2
	}
}

type jsonConfigL2 struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (l *L2) GetKerasLayerConfig() interface{} {
	if l == nil {
		return nil
	}
	return jsonConfigL2{
		ClassName: "L2",
		Config: map[string]interface{}{
			"l2": l.l2,
		},
	}
}

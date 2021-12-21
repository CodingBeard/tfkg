package regularizer

type L1 struct {
	l1 float64
}

func NewL1() *L1 {
	return &L1{
		l1: 0.01,
	}
}

func L1WithL1(l1 float64) func(l *L1) {
	return func(l *L1) {
		l.l1 = l1
	}
}

type jsonConfigL1 struct {
	ClassName string                 `json:"class_name"`
	Name      string                 `json:"name"`
	Config    map[string]interface{} `json:"config"`
}

func (l *L1) GetKerasLayerConfig() interface{} {
	if l == nil {
		return nil
	}
	return jsonConfigL1{
		ClassName: "L1",
		Config: map[string]interface{}{
			"l1": l.l1,
		},
	}
}

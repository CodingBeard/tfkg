package metric

type SparseCategoricalAccuracy struct {
	Name       string
	Confidence float64
	Average    bool
	Precision  int
	total      float64
	count      float64
}

func (m *SparseCategoricalAccuracy) Init() {
	if m.Precision == 0 {
		m.Precision = 4
	}
}

func (m *SparseCategoricalAccuracy) Reset() {
	m.total = 0
	m.count = 0
}

func (m *SparseCategoricalAccuracy) GetName() string {
	return m.Name
}

func (m *SparseCategoricalAccuracy) Compute(yTrue interface{}, yPred interface{}) Value {
	yPredValue := yPred.([][]float32)
	yTrueValue := yTrue.([][]int32)
	correct := 0
	for i, pred := range yPredValue {
		truth := yTrueValue[i][0]
		if float64(pred[truth]) > m.Confidence {
			correct++
		}
	}
	if !m.Average {
		return Value{
			Name:      m.Name,
			Value:     float64(correct) / float64(len(yTrueValue)),
			Precision: m.Precision,
		}
	}

	m.total += float64(correct) / float64(len(yTrueValue))
	m.count++

	return Value{
		Name:      m.Name,
		Value:     m.total / m.count,
		Precision: m.Precision,
	}
}

func (m *SparseCategoricalAccuracy) ComputeFinal() Value {
	return Value{
		Name:      m.Name,
		Value:     m.total / m.count,
		Precision: m.Precision,
	}
}

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
		if float64(pred[int(truth)]) >= m.Confidence {
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

type BinaryAccuracy struct {
	Name       string
	Confidence float64
	Average    bool
	Precision  int
	total      float64
	count      float64
}

func (m *BinaryAccuracy) Init() {
	if m.Precision == 0 {
		m.Precision = 4
	}
}

func (m *BinaryAccuracy) Reset() {
	m.total = 0
	m.count = 0
}

func (m *BinaryAccuracy) GetName() string {
	return m.Name
}

func (m *BinaryAccuracy) Compute(yTrue interface{}, yPred interface{}) Value {
	yPredValue := yPred.([][]float32)
	yTrueValue := yTrue.([][]int32)
	correct := 0
	for i, pred := range yPredValue {
		if float64(pred[0]) >= m.Confidence && yTrueValue[i][0] == 1 {
			correct++
		} else if float64(pred[0]) < m.Confidence && yTrueValue[i][0] == 0 {
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

func (m *BinaryAccuracy) ComputeFinal() Value {
	return Value{
		Name:      m.Name,
		Value:     m.total / m.count,
		Precision: m.Precision,
	}
}

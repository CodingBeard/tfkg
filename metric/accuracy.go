package metric

type SparseCategoricalAccuracy struct {
	Name                    string
	Confidence              float64
	ArgMax                  bool
	Average                 bool
	AverageAcrossCategories bool
	Precision               int
	total                   float64
	count                   float64
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
	if m.AverageAcrossCategories {
		correct := make(map[int]int)
		total := make(map[int]int)
		for i, pred := range yPredValue {
			truth := yTrueValue[i][0]
			count := total[int(truth)]
			count++
			total[int(truth)] = count
			if m.ArgMax {
				var maxValue float32
				maxArg := 0
				for arg, value := range pred {
					if value > maxValue {
						maxValue = value
						maxArg = arg
					}
				}
				if maxArg == int(truth) {
					count := correct[int(truth)]
					count++
					correct[int(truth)] = count
				}
			} else {
				if float64(pred[int(truth)]) >= m.Confidence {
					count := correct[int(truth)]
					count++
					correct[int(truth)] = count
				}
			}
		}
		var averageCorrect float64
		for class, totalCount := range total {
			correctCount := correct[class]
			averageCorrect += float64(correctCount) / float64(totalCount)
		}
		averageCorrect = averageCorrect / float64(len(total))
		if !m.Average {
			return Value{
				Name:      m.Name,
				Value:     averageCorrect,
				Precision: m.Precision,
			}
		}

		m.total += averageCorrect
		m.count++

		return Value{
			Name:      m.Name,
			Value:     m.total / m.count,
			Precision: m.Precision,
		}
	} else {
		correct := 0
		for i, pred := range yPredValue {
			truth := yTrueValue[i][0]
			if m.ArgMax {
				var maxValue float32
				maxArg := 0
				for arg, value := range pred {
					if value > maxValue {
						maxValue = value
						maxArg = arg
					}
				}
				if maxArg == int(truth) {
					correct++
				}
			} else {
				if float64(pred[int(truth)]) >= m.Confidence {
					correct++
				}
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

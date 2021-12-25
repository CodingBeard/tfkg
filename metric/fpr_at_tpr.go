package metric

import (
	"sync"
)

// TODO this only works with two categories 0, 1
type SparseCategoricalFprAtTpr struct {
	Name                 string
	Tpr                  float64
	Attempts             int
	ComputeEveryNBatches int
	Precision            int
	ExtraMetrics         bool
	FinalConfidence      float64
	FinalTpr             float64
	allYTrue             []int32
	allYPred             []float64
	posCount             float64
	negCount             float64
	calls                int
	lastComputedValue    Value
	updateLock           *sync.Mutex
}

func (m *SparseCategoricalFprAtTpr) Init() {
	m.updateLock = &sync.Mutex{}
	if m.Precision == 0 {
		m.Precision = 4
	}
}

func (m *SparseCategoricalFprAtTpr) Reset() {
	m.allYTrue = m.allYTrue[:0]
	m.allYPred = m.allYPred[:0]
	m.posCount = 0
	m.negCount = 0
	m.calls = 0
	m.lastComputedValue = Value{
		Name: m.Name,
	}
}

func (m *SparseCategoricalFprAtTpr) GetName() string {
	return m.Name
}

func (m *SparseCategoricalFprAtTpr) Compute(yTrue interface{}, yPred interface{}) Value {
	yPredValue := yPred.([][]float32)
	yTrueValue := yTrue.([][]int32)

	for offset, yValue := range yTrueValue {
		m.updateLock.Lock()
		if yValue[0] == 0 {
			m.negCount++
		} else {
			m.posCount++
		}
		m.allYTrue = append(m.allYTrue, yValue[0])
		m.allYPred = append(m.allYPred, float64(yPredValue[offset][1]))
		m.updateLock.Unlock()
	}

	m.calls++

	if m.ComputeEveryNBatches > 0 {
		if m.calls%m.ComputeEveryNBatches == 0 {
			m.lastComputedValue = m.ComputeFinal()
		}
	}

	return m.lastComputedValue
}

func (m *SparseCategoricalFprAtTpr) ComputeFinal() Value {
	if m.negCount == 0 || m.posCount == 0 {
		return Value{
			Name: m.Name,
		}
	}

	confRange := []float64{0, 1}

	closest := float64(1)
	closestConfOffset := 0
	closestConfidence := float64(0)
	for i := 0; i < m.Attempts; i++ {
		confidences := make([]float64, m.Attempts)
		for j := 0; j < m.Attempts; j++ {
			confidences[j] = confRange[0] + (((confRange[1] - confRange[0]) / float64(m.Attempts*(i+1))) * float64(j))
		}

		numTps := make([]float64, m.Attempts)
		for offset, yValue := range m.allYTrue {
			if yValue == 1 {
				for confAttempt, confidence := range confidences {
					if m.allYPred[offset] >= confidence {
						numTps[confAttempt]++
					}
				}
			}
		}
		closest = float64(1)
		closestConfOffset = 0
		newDiff := false
		for confOffst, tp := range numTps {
			diff := m.Tpr - (tp / m.posCount)
			if diff <= closest && diff >= 0 {
				closest = diff
				closestConfOffset = confOffst
				closestConfidence = confidences[closestConfOffset]
				newDiff = true
			}
		}
		if !newDiff {
			break
		}
		if closestConfOffset == 0 {
			confRange[0] = confidences[closestConfOffset]
			confRange[1] = confidences[closestConfOffset+1]
		} else if len(confidences)-1 == closestConfOffset {
			confRange[0] = confidences[closestConfOffset-2]
			confRange[1] = confidences[closestConfOffset-1]
		} else {
			confRange[0] = confidences[closestConfOffset-1]
			confRange[1] = confidences[closestConfOffset+1]
		}
	}
	return m.getFpr(closestConfidence)
}

func (m *SparseCategoricalFprAtTpr) getFpr(confidence float64) Value {
	m.FinalConfidence = confidence
	numTp := float64(0)
	numFp := float64(0)
	for offset, yValue := range m.allYTrue {
		if yValue == 1 {
			if m.allYPred[offset] >= confidence {
				numTp++
			}
		} else {
			if m.allYPred[offset] >= confidence {
				numFp++
			}
		}
	}
	m.FinalTpr = numTp / m.posCount
	return Value{
		Name:      m.Name,
		Value:     numFp / m.negCount,
		Precision: m.Precision,
	}
}

func (m *SparseCategoricalFprAtTpr) GetExtraMetrics() []Value {
	if !m.ExtraMetrics {
		return []Value{}
	}
	return []Value{
		{
			Name:      m.Name + "_t",
			Value:     m.FinalTpr,
			Precision: 8,
		},
		{
			Name:      m.Name + "_c",
			Value:     m.FinalConfidence,
			Precision: 8,
		},
	}
}

type BinaryFprAtTpr struct {
	Name                 string
	Tpr                  float64
	Attempts             int
	ComputeEveryNBatches int
	Precision            int
	ExtraMetrics         bool
	FinalConfidence      float64
	FinalTpr             float64
	allYTrue             []int32
	allYPred             []float64
	posCount             float64
	negCount             float64
	calls                int
	lastComputedValue    Value
	updateLock           *sync.Mutex
}

func (m *BinaryFprAtTpr) Init() {
	m.updateLock = &sync.Mutex{}
	if m.Precision == 0 {
		m.Precision = 4
	}
}

func (m *BinaryFprAtTpr) Reset() {
	m.allYTrue = m.allYTrue[:0]
	m.allYPred = m.allYPred[:0]
	m.posCount = 0
	m.negCount = 0
	m.calls = 0
	m.lastComputedValue = Value{
		Name: m.Name,
	}
}

func (m *BinaryFprAtTpr) GetName() string {
	return m.Name
}

func (m *BinaryFprAtTpr) Compute(yTrue interface{}, yPred interface{}) Value {
	yPredValue := yPred.([][]float32)
	yTrueValue := yTrue.([][]int32)

	for offset, yValue := range yTrueValue {
		m.updateLock.Lock()
		if yValue[0] == 0 {
			m.negCount++
		} else {
			m.posCount++
		}
		m.allYTrue = append(m.allYTrue, yValue[0])
		m.allYPred = append(m.allYPred, float64(yPredValue[offset][0]))
		m.updateLock.Unlock()
	}

	m.calls++

	if m.ComputeEveryNBatches > 0 {
		if m.calls%m.ComputeEveryNBatches == 0 {
			m.lastComputedValue = m.ComputeFinal()
		}
	}

	return m.lastComputedValue
}

func (m *BinaryFprAtTpr) ComputeFinal() Value {
	if m.negCount == 0 || m.posCount == 0 {
		return Value{
			Name: m.Name,
		}
	}

	confRange := []float64{0, 1}

	closest := float64(1)
	closestConfOffset := 0
	closestConfidence := float64(0)
	for i := 0; i < m.Attempts; i++ {
		confidences := make([]float64, m.Attempts)
		for j := 0; j < m.Attempts; j++ {
			confidences[j] = confRange[0] + (((confRange[1] - confRange[0]) / float64(m.Attempts*(i+1))) * float64(j))
		}

		numTps := make([]float64, m.Attempts)
		for offset, yValue := range m.allYTrue {
			if yValue == 1 {
				for confAttempt, confidence := range confidences {
					if m.allYPred[offset] >= confidence {
						numTps[confAttempt]++
					}
				}
			}
		}
		closest = float64(1)
		closestConfOffset = 0
		newDiff := false
		for confOffst, tp := range numTps {
			diff := m.Tpr - (tp / m.posCount)
			if diff <= closest && diff >= 0 {
				closest = diff
				closestConfOffset = confOffst
				closestConfidence = confidences[closestConfOffset]
				newDiff = true
			}
		}
		if !newDiff {
			break
		}
		if closestConfOffset == 0 {
			confRange[0] = confidences[closestConfOffset]
			confRange[1] = confidences[closestConfOffset+1]
		} else if len(confidences)-1 == closestConfOffset {
			confRange[0] = confidences[closestConfOffset-2]
			confRange[1] = confidences[closestConfOffset-1]
		} else {
			confRange[0] = confidences[closestConfOffset-1]
			confRange[1] = confidences[closestConfOffset+1]
		}
	}
	return m.getFpr(closestConfidence)
}

func (m *BinaryFprAtTpr) getFpr(confidence float64) Value {
	m.FinalConfidence = confidence
	numTp := float64(0)
	numFp := float64(0)
	for offset, yValue := range m.allYTrue {
		if yValue == 1 {
			if m.allYPred[offset] >= confidence {
				numTp++
			}
		} else {
			if m.allYPred[offset] >= confidence {
				numFp++
			}
		}
	}
	m.FinalTpr = numTp / m.posCount
	return Value{
		Name:      m.Name,
		Value:     numFp / m.negCount,
		Precision: m.Precision,
	}
}

func (m *BinaryFprAtTpr) GetExtraMetrics() []Value {
	if !m.ExtraMetrics {
		return []Value{}
	}
	return []Value{
		{
			Name:      m.Name + "_t",
			Value:     m.FinalTpr,
			Precision: 8,
		},
		{
			Name:      m.Name + "_c",
			Value:     m.FinalConfidence,
			Precision: 8,
		},
	}
}

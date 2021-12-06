package metric

import (
	"sync"
)

type SparseCategoricalTprAtFpr struct {
	Name                 string
	Fpr                  float64
	Attempts             int
	ComputeEveryNBatches int
	Precision            int
	FinalConfidence      float64
	FinalFpr             float64
	ExtraMetrics         bool
	allYTrue             []int32
	allYPred             []float64
	posCount             float64
	negCount             float64
	calls                int
	lastComputedValue    Value
	updateLock           *sync.Mutex
}

func (m *SparseCategoricalTprAtFpr) Init() {
	m.updateLock = &sync.Mutex{}
	if m.Precision == 0 {
		m.Precision = 4
	}
}

func (m *SparseCategoricalTprAtFpr) Reset() {
	m.allYTrue = m.allYTrue[:0]
	m.allYPred = m.allYPred[:0]
	m.posCount = 0
	m.negCount = 0
	m.calls = 0
	m.lastComputedValue = Value{
		Name: m.Name,
	}
}

func (m *SparseCategoricalTprAtFpr) GetName() string {
	return m.Name
}

func (m *SparseCategoricalTprAtFpr) Compute(yTrue interface{}, yPred interface{}) Value {
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

func (m *SparseCategoricalTprAtFpr) ComputeFinal() Value {
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
		confidences := make([]float64, m.Attempts+1)
		confidences[0] = confRange[0]
		confidences[m.Attempts] = confRange[1]
		for j := 1; j < m.Attempts; j++ {
			confidences[j] = confidences[j-1] + ((confRange[1] - confRange[0]) / float64(m.Attempts))
		}

		numFps := make([]float64, m.Attempts+1)
		for offset, yValue := range m.allYTrue {
			if yValue == 0 {
				for confAttempt, confidence := range confidences {
					if m.allYPred[offset] > confidence {
						numFps[confAttempt]++
					}
				}
			}
		}
		closest = float64(1)
		closestConfOffset = 0
		newDiff := false
		for confOffst, fp := range numFps {
			diff := m.Fpr - (fp / m.negCount)
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
			confRange[0] = confidences[closestConfOffset-1]
			confRange[1] = confidences[closestConfOffset]
		} else {
			confRange[0] = confidences[closestConfOffset-1]
			confRange[1] = confidences[closestConfOffset+1]
		}
	}
	return m.getTpr(closestConfidence)
}

func (m *SparseCategoricalTprAtFpr) getTpr(confidence float64) Value {
	m.FinalConfidence = confidence
	numTp := float64(0)
	numFp := float64(0)
	for offset, yValue := range m.allYTrue {
		if yValue == 1 {
			if m.allYPred[offset] > confidence {
				numTp++
			}
		} else {
			if m.allYPred[offset] > confidence {
				numFp++
			}
		}
	}
	m.FinalFpr = numFp / m.negCount
	return Value{
		Name:      m.Name,
		Value:     numTp / m.posCount,
		Precision: m.Precision,
	}
}

func (m *SparseCategoricalTprAtFpr) GetExtraMetrics() []Value {
	if !m.ExtraMetrics {
		return []Value{}
	}
	return []Value{
		{
			Name:      m.Name + "_f",
			Value:     m.FinalFpr,
			Precision: 8,
		},
		{
			Name:      m.Name + "_c",
			Value:     m.FinalConfidence,
			Precision: 8,
		},
	}
}

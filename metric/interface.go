package metric

type Metric interface {
	GetName() string
	Init()
	Reset()
	Compute(yTrue interface{}, yPred interface{}) Value
	ComputeFinal() Value
}

type Value struct {
	Name      string
	Value     float64
	Precision int
}

type HasExtraMetrics interface {
	GetExtraMetrics() []Value
}

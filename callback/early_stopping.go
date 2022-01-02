package callback

import (
	"fmt"
	"strings"
)

type EarlyStoppingOnMetricMode string

var (
	Min EarlyStoppingOnMetricMode = "min"
	Max EarlyStoppingOnMetricMode = "max"
)

type EarlyStoppingOnMetric struct {
	OnEvent    Event
	OnMode     Mode
	MetricName string
	Mode       EarlyStoppingOnMetricMode
	MaxValue   float64
	MinValue   float64
}

func (c *EarlyStoppingOnMetric) Init() error {
	if c.OnEvent == "" {
		return fmt.Errorf("no OnEvent set for callback")
	}
	if c.OnMode == "" {
		return fmt.Errorf("no OnMode set for callback")
	}
	if c.Mode == "" {
		return fmt.Errorf("no Mode set for callback")
	}
	if c.MetricName == "" {
		return fmt.Errorf("unhandled early stopping on metric mode")
	}

	return nil
}

func (c *EarlyStoppingOnMetric) Call(event Event, mode Mode, epoch int, batch int, logs []Log) ([]Action, error) {
	if event != c.OnEvent || mode != c.OnMode {
		return []Action{ActionNop}, nil
	}
	var metricValue float64
	if c.MetricName != "" {
		found := false
		for _, log := range logs {
			if strings.ToLower(log.Name) == strings.ToLower(c.MetricName) {
				metricValue = log.Value
				found = true
			}
		}
		if !found {
			return []Action{ActionNop}, fmt.Errorf("metric %s does not exist for the model", c.MetricName)
		}
	}

	if c.Mode == Min {
		if metricValue <= c.MinValue {
			return []Action{ActionHalt}, nil
		}
	} else if c.Mode == Max {
		if metricValue >= c.MaxValue {
			return []Action{ActionHalt}, nil
		}
	}

	return []Action{ActionNop}, nil
}

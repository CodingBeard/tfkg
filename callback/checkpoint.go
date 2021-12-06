package callback

import (
	"fmt"
	"strings"
)

type CheckpointCompare string

var (
	CheckpointCompareMin CheckpointCompare = "min"
	CheckpointCompareMax CheckpointCompare = "max"
)

type Checkpoint struct {
	OnEvent    Event
	OnMode     Mode
	Loss       bool
	MetricName string
	Compare    CheckpointCompare
	SaveDir    string

	bestValue float64
}

func (c *Checkpoint) GetSaveDir() string {
	return c.SaveDir
}

func (c *Checkpoint) Init() error {
	if c.OnEvent == "" {
		return fmt.Errorf("no OnEvent set for callback")
	}
	if c.OnMode == "" {
		return fmt.Errorf("no OnMode set for callback")
	}
	if c.Compare == "" {
		return fmt.Errorf("no comparison value provided")
	}
	if c.SaveDir == "" {
		return fmt.Errorf("no save dir provided")
	}
	if !c.Loss && c.MetricName == "" {
		return fmt.Errorf("unhandled checkpoint mode")
	}

	return nil
}

func (c *Checkpoint) Call(event Event, mode Mode, epoch int, batch int, logs []Log) ([]Action, error) {
	if event != c.OnEvent || mode != c.OnMode {
		return []Action{ActionNop}, nil
	}
	var metricValue float64
	if c.Loss {
		found := false
		for _, log := range logs {
			if strings.ToLower(log.Name) == "loss" {
				metricValue = log.Value
				found = true
			}
		}
		if !found {
			return []Action{ActionNop}, fmt.Errorf("loss not present in logs")
		}
	} else if c.MetricName != "" {
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

	if c.Compare == CheckpointCompareMin {
		if metricValue < c.bestValue {
			c.bestValue = metricValue
			return []Action{ActionSave}, nil
		}
	} else if c.Compare == CheckpointCompareMax {
		if metricValue > c.bestValue {
			c.bestValue = metricValue
			return []Action{ActionSave}, nil
		}
	}

	return []Action{ActionNop}, nil
}

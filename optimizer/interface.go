package optimizer

import "fmt"

type Optimizer interface {
	GetKerasLayerConfig() interface{}
}

var uniqueNameCounts = make(map[string]int)

func UniqueName(name string) string {
	count := uniqueNameCounts[name]
	count++
	uniqueNameCounts[name] = count

	return fmt.Sprintf("%s_%d", name, count)
}

package util

import (
	"fmt"
	"strings"
	"sync"
)

var nameCounters = map[string]int{}
var nameCounterLock = sync.Mutex{}

func UniqueName(name string) string {
	nameCounterLock.Lock()
	defer nameCounterLock.Unlock()

	count := nameCounters[name]
	count++
	nameCounters[name] = count

	return fmt.Sprintf("%s_%d", strings.TrimRight(name, "_"), count)
}

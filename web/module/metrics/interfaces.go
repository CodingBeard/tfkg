package metrics

import (
	"bytes"
	"fmt"
	"log"
	"runtime"
	"strings"
)

type ErrorHandler interface {
	Error(e error)
}

type defaultErrorHandler struct{}

func (d defaultErrorHandler) Error(e error) {
	buf := make([]byte, 1000000)
	runtime.Stack(buf, false)
	buf = bytes.Trim(buf, "\x00")
	stack := string(buf)
	stackParts := strings.Split(stack, "\n")
	newStackParts := []string{stackParts[0]}
	newStackParts = append(newStackParts, stackParts[3:]...)
	stack = strings.Join(newStackParts, "\n")
	log.Println("ERROR", e.Error()+"\n"+stack)
}

type Logger interface {
	InfoF(category string, message string, args ...interface{})
	DebugF(category string, message string, args ...interface{})
}

type defaultLogger struct{}

func (d defaultLogger) InfoF(category string, message string, args ...interface{}) {
	log.Println(category+":", fmt.Sprintf(message, args...))
}

func (d defaultLogger) DebugF(category string, message string, args ...interface{}) {
	log.Println(category+":", fmt.Sprintf(message, args...))
}

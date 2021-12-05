package util

import (
	"bytes"
	"fmt"
	"runtime"
	"strings"
)

type Error struct {
	e     error
	stack string
}

func NewError(e error) error {
	return &Error{
		e:     e,
		stack: stack(),
	}
}

func (e *Error) Error() string {
	return fmt.Sprintf(
		"ERROR: %s\n%s",
		e.e.Error(),
		e.stack,
	)
}

func stack() string {
	buf := make([]byte, 100000)
	runtime.Stack(buf, false)
	buf = bytes.Trim(buf, "\x00")
	stackParts := strings.Split(string(buf), "\n")
	newStackParts := stackParts[5:]
	return strings.Join(newStackParts, "\n")
}

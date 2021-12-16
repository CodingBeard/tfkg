package common

import (
	"github.com/codingbeard/cbweb/module/cbwebcommon"
)

type Module struct {
	Common       *cbwebcommon.Module
	ErrorHandler ErrorHandler
	Logger       Logger
}

func (m *Module) GetErrorHandler() ErrorHandler {
	if m.ErrorHandler == nil {
		m.ErrorHandler = &defaultErrorHandler{}
	}

	return m.ErrorHandler
}

func (m *Module) GetLogger() Logger {
	if m.Logger == nil {
		m.Logger = &defaultLogger{}
	}

	return m.Logger
}

func (m *Module) GetGlobalTemplates() map[string][]byte {
	return map[string][]byte{}
}

func (m *Module) SetGlobalTemplates(templates map[string][]byte) {

}

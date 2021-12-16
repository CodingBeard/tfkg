package home

import (
	"github.com/codingbeard/cbweb"
	"github.com/codingbeard/cbweb/module/cbwebcommon"
	"github.com/valyala/fasthttp"
)

type Module struct {
	Common       *cbwebcommon.Module
	ErrorHandler ErrorHandler
	Logger       Logger
	NavItems     func(ctx *fasthttp.RequestCtx) []cbweb.NavItem
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

func (m *Module) Home(ctx *fasthttp.RequestCtx) {
	flash := &cbweb.Flash{}

	viewModel := &HomeViewModel{
		NavItems: m.NavItems(ctx),
		Flash:    flash,
		Ctx:      ctx,
	}

	e := m.Common.ExecuteViewModel(ctx, viewModel)

	if e != nil {
		m.GetErrorHandler().Error(e)
		m.Common.GetFiveHundredError()(ctx)
		return
	}
}

func (m *Module) GetGlobalTemplates() map[string][]byte {
	return map[string][]byte{}
}

func (m *Module) SetGlobalTemplates(templates map[string][]byte) {

}

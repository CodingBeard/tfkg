package home

import (
	"github.com/codingbeard/cbweb"
	"github.com/fasthttp/router"
)

var HomeRoute = "/"

func (m *Module) SetRoutes(router *router.Router) {
	router.GET(HomeRoute, cbweb.MiddlewareHandler{}.
		AddMiddleware(cbweb.HtmlMiddleware).
		SetFinal(m.Home).
		Handle,
	)
}

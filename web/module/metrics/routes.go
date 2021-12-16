package metrics

import (
	"github.com/codingbeard/cbweb"
	"github.com/fasthttp/router"
)

var ModelsRoute = "/metrics/models"
var ModelRoute = "/metrics/model"

func (m *Module) SetRoutes(router *router.Router) {
	router.GET(ModelRoute, cbweb.MiddlewareHandler{}.
		AddMiddleware(cbweb.HtmlMiddleware).
		SetFinal(m.Model).
		Handle,
	)
	router.GET(ModelsRoute, cbweb.MiddlewareHandler{}.
		AddMiddleware(cbweb.HtmlMiddleware).
		SetFinal(m.Models).
		Handle,
	)
	router.POST(ModelsRoute, cbweb.MiddlewareHandler{}.
		AddMiddleware(cbweb.HtmlMiddleware).
		SetFinal(m.ModelsPostAction).
		Handle,
	)
}

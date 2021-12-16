package home

import (
	"github.com/codingbeard/cbweb"
	"github.com/valyala/fasthttp"
	"html/template"
)

type HomeViewModel struct {
	Ctx      *fasthttp.RequestCtx
	NavItems []cbweb.NavItem
	Flash    *cbweb.Flash
}

func (t *HomeViewModel) GetTemplates() []string {
	return []string{t.GetMainTemplate()}
}

func (t *HomeViewModel) GetMainTemplate() string {
	return "home/home.gohtml"
}

func (t *HomeViewModel) GetMasterViewModel() cbweb.DefaultMasterViewModel {
	return cbweb.DefaultMasterViewModel{
		ViewIncludes: nil,
		Title:        "TFKG - Home",
		PageTitle:    "Home",
		BodyClasses:  "responsive-nav",
		NavItems:     t.NavItems,
		Path:         template.URL(HomeRoute),
		Flash:        t.Flash,
	}
}

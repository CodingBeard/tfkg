package metrics

import (
	"github.com/codingbeard/cbweb"
	"github.com/codingbeard/cbweb/module/cbwebcommon"
	"github.com/valyala/fasthttp"
	"html/template"
)

type ModelViewModel struct {
	Ctx              *fasthttp.RequestCtx
	NavItems         []cbweb.NavItem
	Flash            *cbweb.Flash
	ModelName        string
	ModelLogs        string
	ModelProgressLog string
	ModelJson        string
	ModelSummary     string
	MetricsViewModel *ModelsViewModel
	MetricsError     error
}

func (t *ModelViewModel) GetTemplates() []string {
	return []string{t.GetMainTemplate()}
}

func (t *ModelViewModel) HasMetricsError() bool {
	return t.MetricsError != nil
}

func (t *ModelViewModel) GetMetricError() string {
	return t.MetricsError.Error()
}

func (t *ModelViewModel) GetDataTable() *cbwebcommon.DataTable {
	return t.MetricsViewModel.GetDataTable()
}

func (t *ModelViewModel) GetLogNames() []string {
	return t.MetricsViewModel.GetLogNames()
}

func (t *ModelViewModel) GetMainTemplate() string {
	return "metrics/model.gohtml"
}

func (t *ModelViewModel) GetMasterViewModel() cbweb.DefaultMasterViewModel {
	return cbweb.DefaultMasterViewModel{
		ViewIncludes: []cbweb.ViewInclude{
			{
				Type: cbweb.ViewIncludeType_JsPostBody,
				Src:  "https://cdn.datatables.net/1.10.15/js/jquery.dataTables.min.js",
			},
			{
				Type: cbweb.ViewIncludeType_JsPostBody,
				Src:  "https://cdn.datatables.net/buttons/1.6.1/js/dataTables.buttons.min.js",
			},
			{
				Type: cbweb.ViewIncludeType_JsPostBody,
				Src:  "https://cdn.datatables.net/buttons/1.6.1/js/buttons.html5.min.js",
			},
			{
				Type: cbweb.ViewIncludeType_JsPostBody,
				Src:  "/js/material-table.js",
			},
		},
		Title:       "TFKG - Model",
		PageTitle:   "Model: " + t.ModelName,
		BodyClasses: "",
		NavItems:    t.NavItems,
		Path:        template.URL(ModelsRoute),
		Flash:       t.Flash,
	}
}

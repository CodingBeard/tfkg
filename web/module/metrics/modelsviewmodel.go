package metrics

import (
	"github.com/codingbeard/cbweb"
	"github.com/codingbeard/cbweb/module/cbwebcommon"
	"github.com/valyala/fasthttp"
	"html/template"
	"sort"
)

type ModelsViewModel struct {
	Ctx      *fasthttp.RequestCtx
	NavItems []cbweb.NavItem
	Flash    *cbweb.Flash
	Metrics  []Record
	AllRows  bool
}

func (t *ModelsViewModel) GetTemplates() []string {
	return []string{t.GetMainTemplate()}
}

func (t *ModelsViewModel) GetMainTemplate() string {
	return "metrics/models.gohtml"
}

func (t *ModelsViewModel) GetMetricsColumns() map[string]int {
	allLogNames := map[string]int{
		"model":    0,
		"datetime": 1,
		"event":    2,
		"mode":     3,
		"epoch":    4,
		"batch":    5,
	}
	uniqueMetricCount := 6
	for _, metric := range t.Metrics {

		for _, row := range metric.Rows {
			for _, metricName := range row.LogNames {
				if _, ok := allLogNames[metricName]; !ok {
					allLogNames[metricName] = uniqueMetricCount
					uniqueMetricCount++
				}
			}
		}
	}
	return allLogNames
}

func (t *ModelsViewModel) GetLogNames() []string {
	allLogNames := map[string]int{}
	uniqueMetricCount := 6
	for _, metric := range t.Metrics {

		for _, row := range metric.Rows {
			for _, metricName := range row.LogNames {
				if _, ok := allLogNames[metricName]; !ok {
					allLogNames[metricName] = uniqueMetricCount
					uniqueMetricCount++
				}
			}
		}
	}
	type kv struct {
		k string
		v int
	}
	var kvs []kv
	for key, value := range allLogNames {
		kvs = append(kvs, kv{
			k: key,
			v: value,
		})
	}
	sort.Slice(kvs, func(i, j int) bool {
		return kvs[i].k < kvs[j].k
	})
	var columns []string
	for _, k := range kvs {
		columns = append(columns, k.k)
	}

	return columns
}

func (t *ModelsViewModel) GetDataTable() *cbwebcommon.DataTable {
	allLogNames := t.GetMetricsColumns()

	sortedLogNames := make(map[string]int)
	type kv struct {
		k string
		v int
	}
	var kvs []kv
	columns := []string{
		"model",
		"datetime",
		"event",
		"mode",
		"epoch",
		"batch",
	}
	for key, value := range allLogNames {
		found := false
		for _, column := range columns {
			if key == column {
				found = true
			}
		}
		if found {
			continue
		}
		kvs = append(kvs, kv{
			k: key,
			v: value,
		})
	}
	sort.Slice(kvs, func(i, j int) bool {
		return kvs[i].k < kvs[j].k
	})
	for _, k := range kvs {
		columns = append(columns, k.k)
	}
	for i, column := range columns {
		sortedLogNames[column] = i
	}

	var data [][]interface{}
	for _, metric := range t.Metrics {

		lastRows := make(map[string][]interface{})
		for _, row := range metric.Rows {
			rowData := make([]interface{}, len(allLogNames))
			rowData[0] = metric.ModelName
			rowData[1] = row.Datetime
			rowData[2] = row.Event
			rowData[3] = row.Mode
			rowData[4] = row.Epoch
			rowData[5] = row.Batch
			for offset, name := range row.LogNames {
				rowData[sortedLogNames[name]] = row.Logs[offset]
			}
			if t.AllRows {
				data = append(data, rowData)
			} else {
				lastRows[row.Event+row.Mode] = rowData
			}
		}
		if !t.AllRows {
			for _, rowData := range lastRows {
				data = append(data, rowData)
			}
		}
	}

	var dtColumns []cbwebcommon.DataTableColumn

	for _, column := range columns {
		filterable := false
		if column == "event" || column == "mode" {
			filterable = true
		}
		dtColumns = append(dtColumns, cbwebcommon.DataTableColumn{
			Title:        column,
			EditableName: column,
			Filterable:   filterable,
		})
	}

	return &cbwebcommon.DataTable{
		TableId:           "modelmetrics",
		Columns:           dtColumns,
		Data:              data,
		GroupByColumn:     true,
		GroupColumnOffset: 0,
	}
}

func (t *ModelsViewModel) GetMasterViewModel() cbweb.DefaultMasterViewModel {
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
		Title:       "TFKG - Metrics",
		PageTitle:   "Metrics",
		BodyClasses: "",
		NavItems:    t.NavItems,
		Path:        template.URL(ModelsRoute),
		Flash:       t.Flash,
	}
}

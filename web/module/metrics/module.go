package metrics

import (
	"encoding/csv"
	"errors"
	"github.com/codingbeard/cbutil"
	"github.com/codingbeard/cbweb"
	"github.com/codingbeard/cbweb/module/cbwebcommon"
	"github.com/karrick/godirwalk"
	"github.com/valyala/fasthttp"
	"io"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type Module struct {
	Common        *cbwebcommon.Module
	ErrorHandler  ErrorHandler
	Logger        Logger
	NavItems      func(ctx *fasthttp.RequestCtx) []cbweb.NavItem
	IgnoredModels []string
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

func (m *Module) Model(ctx *fasthttp.RequestCtx) {
	flash := &cbweb.Flash{}

	modelName := string(ctx.URI().QueryArgs().Peek("modelName"))

	logFileBytes, e := ioutil.ReadFile(filepath.Join("/go/src/tfkg/logs/", modelName, "training.log"))
	if e != nil {
		logFileBytes = []byte("No logs found at: " + filepath.Join("/go/src/tfkg/logs/", modelName, "training.log"))
	}
	progressFileBytes, _ := ioutil.ReadFile(filepath.Join("/go/src/tfkg/logs/", modelName, "progress.log"))

	modelJsonBytes, _ := ioutil.ReadFile(filepath.Join("/go/src/tfkg/logs/", modelName, "model.json"))
	modelSummaryBytes, _ := ioutil.ReadFile(filepath.Join("/go/src/tfkg/logs/", modelName, "model-summary.txt"))

	var recordsError error
	var records []Record
	records, recordsError = m.getMetrics(modelName)

	metricsViewModel := &ModelsViewModel{
		Metrics: records,
		AllRows: true,
	}

	viewModel := &ModelViewModel{
		NavItems:         m.NavItems(ctx),
		Flash:            flash,
		Ctx:              ctx,
		ModelName:        modelName,
		ModelLogs:        string(logFileBytes),
		ModelProgressLog: string(progressFileBytes),
		ModelJson:        string(modelJsonBytes),
		ModelSummary:     string(modelSummaryBytes),
		MetricsViewModel: metricsViewModel,
		MetricsError:     recordsError,
	}

	e = m.Common.ExecuteViewModel(ctx, viewModel)

	if e != nil {
		m.GetErrorHandler().Error(e)
		m.Common.GetFiveHundredError()(ctx)
		return
	}
}

func (m *Module) getMetrics(modelName string) ([]Record, error) {
	var records []Record

	requiredHeaders := []string{
		"datetime",
		"event",
		"mode",
		"epoch",
		"batch",
	}
	stat, e := os.Stat(filepath.Join("/go/src/tfkg/logs", modelName, "training.log"))
	if e == nil {
		records = append(records, Record{
			ModelName: modelName,
			DateTime:  stat.ModTime().Format(cbutil.DateTimeFormat),
		})
	}
	e = godirwalk.Walk(filepath.Join("/go/src/tfkg/logs", modelName), &godirwalk.Options{
		Callback: func(osPathname string, directoryEntry *godirwalk.Dirent) error {
			if !strings.HasSuffix(osPathname, ".csv") {
				return nil
			}
			pathParts := strings.Split(osPathname, "/")
			if len(pathParts) < 2 {
				return nil
			}

			modelRecord := Record{
				ModelName: pathParts[len(pathParts)-2],
				Rows:      nil,
			}
			file, e := os.Open(osPathname)
			if e != nil {
				m.ErrorHandler.Error(e)
				return e
			}
			reader := csv.NewReader(file)
			headers, e := reader.Read()
			if e != nil && !errors.Is(e, io.EOF) {
				m.ErrorHandler.Error(e)
				return e
			}

			foundRequiredHeaders := 0
			for _, requiredHeader := range requiredHeaders {
				for _, header := range headers {
					if strings.ToLower(requiredHeader) == strings.ToLower(header) {
						foundRequiredHeaders++
					}
				}
			}
			if foundRequiredHeaders != len(requiredHeaders) {
				return nil
			}

			headersNameMap := make(map[string]int)
			for offset, header := range headers {
				headersNameMap[header] = offset
			}

			headersOffsetMap := make(map[int]string)
			for offset, header := range headers {
				headersOffsetMap[offset] = header
			}

			for true {
				line, e := reader.Read()
				if errors.Is(e, io.EOF) {
					break
				} else if e != nil {
					m.ErrorHandler.Error(e)
					return e
				}

				epoch, e := strconv.Atoi(line[headersNameMap["epoch"]])
				if e != nil {
					m.ErrorHandler.Error(e)
					return e
				}

				batch, e := strconv.Atoi(line[headersNameMap["batch"]])
				if e != nil {
					m.ErrorHandler.Error(e)
					return e
				}

				var logNames []string
				var logs []float32

				for i, metric := range line {
					if headersOffsetMap[i] == "datetime" ||
						headersOffsetMap[i] == "event" ||
						headersOffsetMap[i] == "mode" ||
						headersOffsetMap[i] == "epoch" ||
						headersOffsetMap[i] == "batch" {
						continue
					}
					value, e := strconv.ParseFloat(metric, 32)
					if e != nil {
						continue
					}
					if math.IsNaN(value) {
						continue
					}
					logNames = append(logNames, headersOffsetMap[i])
					logs = append(logs, float32(value))
				}

				modelRecord.Rows = append(modelRecord.Rows, RecordRow{
					Datetime: line[headersNameMap["datetime"]],
					Event:    line[headersNameMap["event"]],
					Mode:     line[headersNameMap["mode"]],
					Epoch:    epoch,
					Batch:    batch,
					LogNames: logNames,
					Logs:     logs,
				})
			}

			records = append(records, modelRecord)

			return nil
		},
	})
	if e != nil {
		return nil, e
	}
	return records, nil
}

type RecordRow struct {
	Datetime string
	Event    string
	Mode     string
	Epoch    int
	Batch    int
	LogNames []string
	Logs     []float32
}

type Record struct {
	ModelName string
	DateTime  string
	Rows      []RecordRow
}

func (m *Module) Models(ctx *fasthttp.RequestCtx) {
	m.loadIgnoredModels()
	var records []Record
	flash := &cbweb.Flash{}

	modelDirs, e := filepath.Glob("/go/src/tfkg/logs/*")
	if e != nil {
		m.ErrorHandler.Error(e)
		m.Common.GetFiveHundredError()(ctx)
		return
	}

	for _, dir := range modelDirs {
		stat, e := os.Stat(dir)
		if e != nil {
			continue
		}
		if !stat.IsDir() {
			continue
		}
		_, finalDir := filepath.Split(dir)
		if m.isIgnoredModel(finalDir) {
			continue
		}
		modelRecords, e := m.getMetrics(finalDir)
		if e != nil {
			m.ErrorHandler.Error(e)
			continue
		}
		records = append(records, modelRecords...)
	}

	viewModel := &ModelsViewModel{
		NavItems: m.NavItems(ctx),
		Flash:    flash,
		Ctx:      ctx,
		Metrics:  records,
		AllRows:  false,
	}

	e = m.Common.ExecuteViewModel(ctx, viewModel)

	if e != nil {
		m.GetErrorHandler().Error(e)
		m.Common.GetFiveHundredError()(ctx)
		return
	}
}

func (m *Module) ModelsPostAction(ctx *fasthttp.RequestCtx) {
	if !ctx.IsPost() {
		return
	}
	postArgs := ctx.PostArgs()
	if postArgs.Has("hide-model") && postArgs.Has("model-name") {
		m.IgnoredModels = append(m.IgnoredModels, string(postArgs.Peek("model-name")))
		m.saveIgnoredModels()
	}
	ctx.Redirect(ctx.URI().String(), 302)
}

func (m *Module) isIgnoredModel(modelName string) bool {
	for _, ignoredModel := range m.IgnoredModels {
		if strings.ToLower(ignoredModel) == strings.ToLower(modelName) {
			return true
		}
	}
	return false
}

func (m *Module) saveIgnoredModels() {
	file, e := os.Create("/go/src/tfkg/logs/ignored-models.csv")
	if e != nil {
		m.ErrorHandler.Error(e)
		return
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	for _, ignoredModel := range m.IgnoredModels {
		e = writer.Write([]string{ignoredModel})
		if e != nil {
			m.ErrorHandler.Error(e)
			return
		}
		writer.Flush()
	}
}

func (m *Module) loadIgnoredModels() {
	file, e := os.Open("/go/src/tfkg/logs/ignored-models.csv")
	if errors.Is(e, os.ErrNotExist) {
		return
	} else if e != nil {
		m.ErrorHandler.Error(e)
		return
	}
	m.IgnoredModels = m.IgnoredModels[:0]
	reader := csv.NewReader(file)
	for true {
		line, e := reader.Read()
		if errors.Is(e, io.EOF) {
			break
		} else if e != nil {
			m.ErrorHandler.Error(e)
			return
		}
		m.IgnoredModels = append(m.IgnoredModels, line[0])
	}
}

func (m *Module) GetGlobalTemplates() map[string][]byte {
	return map[string][]byte{}
}

func (m *Module) SetGlobalTemplates(templates map[string][]byte) {

}

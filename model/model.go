package model

//go:generate go run ../generate/generate.go

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/codingbeard/cberrors"
	"github.com/codingbeard/cblog"
	"github.com/codingbeard/tfkg/callback"
	"github.com/codingbeard/tfkg/data"
	"github.com/codingbeard/tfkg/layer"
	"github.com/codingbeard/tfkg/metric"
	"github.com/codingbeard/tfkg/optimizer"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

type Loss string

var (
	tempModelDir                           = "tfkg_temp_model"
	LossBinaryCrossentropy            Loss = "binary_crossentropy"
	LossSparseCategoricalCrossentropy Loss = "sparse_categorical_crossentropy"
)

type TfkgModel struct {
	model                  *tf.SavedModel
	layers                 []layer.Layer
	isSequential           bool
	pbCache                []byte
	modelDefinitionSaveDir string

	errorHandler *cberrors.ErrorsContainer
	logger       *cblog.Logger
}

func NewSequentialModel(
	logger *cblog.Logger,
	errorHandler *cberrors.ErrorsContainer,
	input layer.Layer,
	layers ...layer.Layer,
) *TfkgModel {
	layersWithInputs := []layer.Layer{
		input,
	}
	lastLayer := input
	for _, l := range layers {
		l.SetInputs(lastLayer)
		layersWithInputs = append(layersWithInputs, l)
		lastLayer = l
	}
	return &TfkgModel{
		isSequential: true,
		layers:       layersWithInputs,
		errorHandler: errorHandler,
		logger:       logger,
	}
}

func NewModel(
	logger *cblog.Logger,
	errorHandler *cberrors.ErrorsContainer,
	output layer.Layer,
) *TfkgModel {
	return &TfkgModel{
		isSequential: false,
		layers:       getPreviousLayers(output, []layer.Layer{}),
		errorHandler: errorHandler,
		logger:       logger,
	}
}

func getPreviousLayers(l layer.Layer, layers []layer.Layer) []layer.Layer {
	if len(l.GetInputs()) > 0 {
		for _, input := range l.GetInputs() {
			layers = getPreviousLayers(input, layers)
		}
	}

	found := false
	for _, existingLayer := range layers {
		if existingLayer.GetName() == l.GetName() {
			found = true
			break
		}
	}
	if !found {
		layers = append(layers, l)
	}

	return layers
}

func LoadModel(
	errorHandler *cberrors.ErrorsContainer,
	logger *cblog.Logger,
	dir string,
) (*TfkgModel, error) {
	m, e := tf.LoadSavedModel(dir, []string{"serve"}, nil)
	if e != nil {
		errorHandler.Error(e)
		return nil, e
	}

	pbCache, e := ioutil.ReadFile(filepath.Join(dir, "saved_model.pb"))
	if e != nil {
		errorHandler.Error(e)
		return nil, e
	}

	// TODO: verify that the loaded model is in fact a compatible tfkg model

	return &TfkgModel{
		model:                  m,
		pbCache:                pbCache,
		errorHandler:           errorHandler,
		logger:                 logger,
		modelDefinitionSaveDir: dir,
	}, nil
}

func (m *TfkgModel) GetWeights() ([]*tf.Tensor, error) {
	var variableOutputs []tf.Output
	output := 0
	for _, info := range m.model.Signatures["get_weights"].Outputs {
		parts := strings.Split(info.Name, ":")
		if len(parts) != 2 {
			e := fmt.Errorf("error getting output for get_weights signature in GetWeights")
			m.errorHandler.Error(e)
			return nil, e
		}
		name := parts[0]
		variableOutputs = append(variableOutputs, m.model.Graph.Operation(name).Output(output))
		output++
	}

	results, e := m.model.Session.Run(
		map[tf.Output]*tf.Tensor{},
		variableOutputs,
		nil,
	)
	if e != nil {
		m.errorHandler.Error(e)
		return nil, e
	}

	return results, nil
}

func (m *TfkgModel) GetNamedWeights() (map[string]*tf.Tensor, error) {
	weightNamesFile, e := ioutil.ReadFile(filepath.Join(m.modelDefinitionSaveDir, "weight_names.json"))
	if e != nil {
		m.errorHandler.Error(e)
		return nil, e
	}

	var names []string
	e = json.Unmarshal(weightNamesFile, &names)

	results, e := m.GetWeights()
	if e != nil {
		return nil, e
	}

	namedResults := make(map[string]*tf.Tensor)
	for i, name := range names {
		if i >= len(results) {
			e = fmt.Errorf("there were not enough weights in the call to m.GetWeights for the saved weight names")
			m.errorHandler.Error(e)
			return nil, e
		}
		namedResults[name] = results[i]
	}

	return namedResults, nil
}

func (m *TfkgModel) Predict(inputs ...*tf.Tensor) (*tf.Tensor, error) {
	if len(inputs) < 1 {
		e := fmt.Errorf("no inputs provided")
		m.errorHandler.Error(e)
		return nil, e
	}
	var predictOutputs []tf.Output
	output := 0
	for _, info := range m.model.Signatures["predict"].Outputs {
		parts := strings.Split(info.Name, ":")
		if len(parts) != 2 {
			e := fmt.Errorf("error getting output for predict signature in fit")
			m.errorHandler.Error(e)
			return nil, e
		}
		name := parts[0]
		predictOutputs = append(predictOutputs, m.model.Graph.Operation(name).Output(output))
		output++
	}
	predictInputs := map[tf.Output]*tf.Tensor{}
	for i, inputTensor := range inputs {
		predictInputs[m.model.Graph.Operation(fmt.Sprintf("predict_inputs_%d", i)).Output(0)] = inputTensor
	}

	results, e := m.model.Session.Run(
		predictInputs,
		predictOutputs,
		nil,
	)
	if e != nil {
		m.errorHandler.Error(e)
		return nil, e
	}

	return results[0], nil
}

type FitConfig struct {
	Epochs     int
	BatchSize  int
	Validation bool
	PreFetch   int
	Metrics    []metric.Metric
	Callbacks  []callback.Callback
	Verbose    int
}

func (m *TfkgModel) Fit(
	dataset data.Dataset,
	config FitConfig,
) {
	trainSig := m.model.Signatures["learn"]
	var trainOutputs []tf.Output
	output := 0
	for _, info := range trainSig.Outputs {
		parts := strings.Split(info.Name, ":")
		if len(parts) != 2 {
			e := fmt.Errorf("error getting output for train signature in fit")
			m.errorHandler.Error(e)
			return
		}
		name := parts[0]
		trainOutputs = append(trainOutputs, m.model.Graph.Operation(name).Output(output))
		output++
	}

	if config.Epochs == 0 {
		config.Epochs = 1
	}

	for i := range config.Metrics {
		config.Metrics[i].Init()
	}
	for i := range config.Callbacks {
		e := config.Callbacks[i].Init()
		if e != nil {
			m.errorHandler.Error(e)
			return
		}
	}

	for epoch := 1; epoch <= config.Epochs; epoch++ {

		generatorChan := dataset.
			SetMode(data.GeneratorModeTrain).
			GeneratorChan(config.BatchSize, config.PreFetch)

		labelOp := m.model.Graph.Operation(fmt.Sprintf("learn_%s", "y")).Output(0)
		classWeightOp := m.model.Graph.Operation(fmt.Sprintf("learn_%s", "class_weights")).Output(0)

		var inputOps []tf.Output
		for offset := range dataset.GetColumnNames() {
			inputOps = append(inputOps, m.model.Graph.Operation(fmt.Sprintf("learn_inputs_%d", offset)).Output(0))
		}

		halt := false

		var trainLogs []callback.Log

		batch := 1
		totalBatches := dataset.Len() / config.BatchSize
		trainTotalLoss := float64(0)
		for generatorBatch := range generatorChan {
			if halt {
				break
			}

			x, y, classWeights := generatorBatch.X, generatorBatch.Y, generatorBatch.ClassWeights

			inputs := map[tf.Output]*tf.Tensor{
				labelOp:       y,
				classWeightOp: classWeights,
			}

			for offset, op := range inputOps {
				inputs[op] = x[offset]
			}

			result, e := m.model.Session.Run(
				inputs,
				trainOutputs,
				nil,
			)
			if e != nil {
				m.errorHandler.Error(e)
				return
			}

			loss := result[0].Value().(float32)
			yTrue := y.Value().([][]int32)
			yPred := result[1].Value().([][]float32)

			trainTotalLoss += float64(loss)

			trainLogs = []callback.Log{
				{
					Name:  callback.LoggerVerbose,
					Value: float64(config.Verbose),
				},
				{
					Name:  callback.LoggerTotalBatches,
					Value: float64(totalBatches),
				},
				{
					Name:  callback.LoggerPrefetched,
					Value: float64(len(generatorChan)),
				},
				{
					Name:      "loss",
					Value:     trainTotalLoss / float64(batch),
					Precision: 4,
				},
			}

			event := callback.EventDuring
			if batch == 1 {
				event = callback.EventStart
			} else if batch == totalBatches {
				event = callback.EventEnd
			}

			trainLogs = append(trainLogs, m.getMetricLogs(
				config.Metrics,
				"",
				batch == totalBatches,
				yTrue,
				yPred,
			)...)

			halt, _ = m.processCallbacks(
				config.Callbacks,
				event,
				callback.ModeTrain,
				epoch,
				batch,
				trainLogs,
			)

			batch++
		}
		for i := range config.Metrics {
			config.Metrics[i].Reset()
		}
		if config.Validation {
			valSig := m.model.Signatures["evaluate"]
			output := 0
			var valOutputs []tf.Output
			for _, info := range valSig.Outputs {
				parts := strings.Split(info.Name, ":")
				if len(parts) != 2 {
					e := fmt.Errorf("error getting output for val signature in fit")
					m.errorHandler.Error(e)
					return
				}
				name := parts[0]
				valOutputs = append(valOutputs, m.model.Graph.Operation(name).Output(output))
				output++
			}

			generatorChan := dataset.
				SetMode(data.GeneratorModeVal).
				GeneratorChan(config.BatchSize, config.PreFetch)

			valLabelOp := m.model.Graph.Operation(fmt.Sprintf("evaluate_%s", "y")).Output(0)
			classWeightOp := m.model.Graph.Operation(fmt.Sprintf("evaluate_%s", "class_weights")).Output(0)

			var valInputOps []tf.Output
			for offset := range dataset.GetColumnNames() {
				valInputOps = append(valInputOps, m.model.Graph.Operation(fmt.Sprintf("evaluate_inputs_%d", offset)).Output(0))
			}

			batch = 1
			totalBatches := dataset.Len() / config.BatchSize
			valTotalLoss := float64(0)
			for generatorBatch := range generatorChan {
				if halt {
					break
				}

				x, y, classWeights := generatorBatch.X, generatorBatch.Y, generatorBatch.ClassWeights

				inputs := map[tf.Output]*tf.Tensor{
					valLabelOp:    y,
					classWeightOp: classWeights,
				}

				for offset, op := range valInputOps {
					inputs[op] = x[offset]
				}

				result, e := m.model.Session.Run(
					inputs,
					valOutputs,
					nil,
				)
				if e != nil {
					m.errorHandler.Error(e)
					return
				}

				yTrue := y.Value().([][]int32)
				loss := result[0].Value().(float32)
				yPred := result[1].Value().([][]float32)

				valTotalLoss += float64(loss)

				valLogs := []callback.Log{
					{
						Name:  callback.LoggerPrefetched,
						Value: float64(len(generatorChan)),
					},
					{
						Name:  callback.LoggerTotalBatches,
						Value: float64(totalBatches),
					},
					{
						Name:      "val_loss",
						Value:     valTotalLoss / float64(batch),
						Precision: 4,
					},
				}

				event := callback.EventDuring
				if batch == 1 {
					event = callback.EventStart
				} else if batch == totalBatches {
					event = callback.EventEnd
				}

				valLogs = append(valLogs, m.getMetricLogs(
					config.Metrics,
					"val_",
					batch == totalBatches,
					yTrue,
					yPred,
				)...)

				halt, _ = m.processCallbacks(
					config.Callbacks,
					event,
					callback.ModeVal,
					epoch,
					batch,
					append(trainLogs, valLogs...),
				)

				batch++
			}
			for i := range config.Metrics {
				config.Metrics[i].Reset()
			}
		}
	}
}

type EvaluateConfig struct {
	BatchSize int
	PreFetch  int
	Metrics   []metric.Metric
	Callbacks []callback.Callback
	Verbose   int
}

func (m *TfkgModel) Evaluate(
	mode data.GeneratorMode,
	dataset data.Dataset,
	config EvaluateConfig,
) {

	for i := range config.Metrics {
		config.Metrics[i].Init()
	}
	for i := range config.Callbacks {
		e := config.Callbacks[i].Init()
		if e != nil {
			m.errorHandler.Error(e)
			return
		}
	}
	evaluateSig := m.model.Signatures["evaluate"]
	output := 0
	var evaluateOutputs []tf.Output
	for _, info := range evaluateSig.Outputs {
		parts := strings.Split(info.Name, ":")
		if len(parts) != 2 {
			e := fmt.Errorf("error getting output for evaluate signature in fit")
			m.errorHandler.Error(e)
			return
		}
		name := parts[0]
		evaluateOutputs = append(evaluateOutputs, m.model.Graph.Operation(name).Output(output))
		output++
	}

	generatorChan := dataset.
		SetMode(mode).
		GeneratorChan(config.BatchSize, config.PreFetch)

	evaluateLabelOp := m.model.Graph.Operation(fmt.Sprintf("evaluate_%s", "y")).Output(0)
	classWeightOp := m.model.Graph.Operation(fmt.Sprintf("evaluate_%s", "class_weights")).Output(0)

	var evaluateInputOps []tf.Output
	for offset := range dataset.GetColumnNames() {
		evaluateInputOps = append(evaluateInputOps, m.model.Graph.Operation(fmt.Sprintf("evaluate_inputs_%d", offset)).Output(0))
	}

	batch := 1
	totalBatches := dataset.Len() / config.BatchSize
	evaluateTotalLoss := float64(0)
	var halt bool
	for generatorBatch := range generatorChan {
		if halt {
			break
		}

		x, y, classWeights := generatorBatch.X, generatorBatch.Y, generatorBatch.ClassWeights

		inputs := map[tf.Output]*tf.Tensor{
			evaluateLabelOp: y,
			classWeightOp:   classWeights,
		}

		for offset, op := range evaluateInputOps {
			inputs[op] = x[offset]
		}

		result, e := m.model.Session.Run(
			inputs,
			evaluateOutputs,
			nil,
		)
		if e != nil {
			m.errorHandler.Error(e)
			return
		}

		yTrue := y.Value().([][]int32)
		loss := result[0].Value().(float32)
		yPred := result[1].Value().([][]float32)

		evaluateTotalLoss += float64(loss)

		evaluateLogs := []callback.Log{
			{
				Name:  callback.LoggerVerbose,
				Value: float64(config.Verbose),
			},
			{
				Name:  callback.LoggerPrefetched,
				Value: float64(len(generatorChan)),
			},
			{
				Name:  callback.LoggerTotalBatches,
				Value: float64(totalBatches),
			},
			{
				Name:      string(mode) + "_loss",
				Value:     evaluateTotalLoss / float64(batch),
				Precision: 4,
			},
		}

		event := callback.EventDuring
		if batch == 1 {
			event = callback.EventStart
		} else if batch == totalBatches {
			event = callback.EventEnd
		}

		callbackMode := callback.ModeTrain
		if mode == data.GeneratorModeVal {
			callbackMode = callback.ModeVal
		} else if mode == data.GeneratorModeTest {
			callbackMode = callback.ModeTest
		}

		evaluateLogs = append(evaluateLogs, m.getMetricLogs(
			config.Metrics,
			string(mode)+"_",
			batch == totalBatches,
			yTrue,
			yPred,
		)...)

		halt, _ = m.processCallbacks(
			config.Callbacks,
			event,
			callbackMode,
			1,
			batch,
			evaluateLogs,
		)

		batch++
	}
}

func (m *TfkgModel) getMetricLogs(
	metrics []metric.Metric,
	namePrefix string,
	isLastBatch bool,
	yTrue interface{},
	yPred interface{},
) []callback.Log {
	var logs []callback.Log
	for _, met := range metrics {
		if isLastBatch {
			met.Compute(yTrue, yPred)
			final := met.ComputeFinal()

			logs = append(logs, callback.Log{
				Name:      namePrefix + final.Name,
				Value:     final.Value,
				Precision: final.Precision,
			})

			if metWithExtra, ok := met.(metric.HasExtraMetrics); ok {
				for _, extraMetric := range metWithExtra.GetExtraMetrics() {
					logs = append(logs, callback.Log{
						Name:      namePrefix + extraMetric.Name,
						Value:     extraMetric.Value,
						Precision: extraMetric.Precision,
					})
				}
			}
		} else {
			value := met.Compute(yTrue, yPred)
			logs = append(logs, callback.Log{
				Name:      namePrefix + met.GetName(),
				Value:     value.Value,
				Precision: value.Precision,
			})
			if metWithExtra, ok := met.(metric.HasExtraMetrics); ok {
				for _, extraMetric := range metWithExtra.GetExtraMetrics() {
					logs = append(logs, callback.Log{
						Name:      namePrefix + extraMetric.Name,
						Value:     extraMetric.Value,
						Precision: extraMetric.Precision,
					})
				}
			}
		}
	}

	return logs
}

func (m *TfkgModel) processCallbacks(
	callbacks []callback.Callback,
	event callback.Event,
	mode callback.Mode,
	epoch int,
	batch int,
	logs []callback.Log,
) (halt bool, saved bool) {
	halt, saved = m.callCallbacks(
		callbacks,
		event,
		mode,
		epoch,
		batch,
		logs,
	)

	if saved {
		halt, _ = m.callCallbacks(
			callbacks,
			callback.EventSave,
			mode,
			epoch,
			batch,
			logs,
		)
	}

	return halt, saved
}

func (m *TfkgModel) callCallbacks(
	callbacks []callback.Callback,
	event callback.Event,
	mode callback.Mode,
	epoch int,
	batch int,
	metrics []callback.Log,
) (halt bool, saved bool) {
	for _, call := range callbacks {
		actions, e := call.Call(event, mode, epoch, batch, metrics)
		if e != nil {
			m.errorHandler.Error(e)
			continue
		}

		for _, action := range actions {
			if action == callback.ActionSave {
				saveDirGetter, ok := call.(callback.HasSaveDir)
				if ok {
					e = m.Save(saveDirGetter.GetSaveDir())
					if e != nil {
						m.errorHandler.Error(e)
					} else {
						saved = true
					}
				}
			} else if action == callback.ActionHalt {
				halt = true
			}
		}
	}

	return halt, saved
}

func (m *TfkgModel) Save(dir string) error {
	signatureCount := len(m.model.Signatures)

	e := os.MkdirAll(filepath.Join(dir, "variables"), os.ModePerm)
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}

	e = ioutil.WriteFile(filepath.Join(dir, "saved_model.pb"), m.pbCache, os.ModePerm)
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}

	filenameInput, e := tf.NewTensor(filepath.Join(dir, "variables/variables"))
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}

	_, e = m.model.Session.Run(
		map[tf.Output]*tf.Tensor{
			m.model.Graph.Operation("saver_filename").Output(0): filenameInput,
		},
		[]tf.Output{
			m.model.Graph.Operation(fmt.Sprintf("StatefulPartitionedCall_%d", signatureCount-1)).Output(0),
		},
		nil,
	)
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}
	return nil
}

type pythonConfig struct {
	ModelConfig            string                 `json:"model_config"`
	SaveDir                string                 `json:"save_dir"`
	ModelDefinitionSaveDir string                 `json:"model_definition_save_dir"`
	Loss                   string                 `json:"loss"`
	Optimizer              interface{}            `json:"optimizer"`
	Weights                map[string]interface{} `json:"weights"`
}

func (m *TfkgModel) CompileAndLoad(loss Loss, optimizer optimizer.Optimizer, modelDefinitionSaveDir string) error {
	m.logger.InfoF("model", "Compiling and loading model. If anything goes wrong python error messages will be printed out.")
	m.modelDefinitionSaveDir = modelDefinitionSaveDir
	modelConfig, e := m.generateKerasDefinitionJson()
	if e != nil {
		return e
	}

	tempDir := filepath.Join(os.TempDir(), "/tfkg")

	e = os.MkdirAll(tempDir, os.ModePerm)
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}

	if modelDefinitionSaveDir != "" {
		indentedJson := bytes.NewBuffer([]byte{})
		e = json.Indent(indentedJson, []byte(modelConfig), "", "  ")
		if e != nil {
			m.errorHandler.Error(e)
			return e
		}
		e = ioutil.WriteFile(filepath.Join(modelDefinitionSaveDir, "model.json"), indentedJson.Bytes(), os.ModePerm)
		if e != nil {
			m.errorHandler.Error(e)
			return e
		}
	}
	weights := make(map[string]interface{})
	for _, l := range m.layers {
		w := l.GetLayerWeights()
		if w != nil {
			weights[l.GetName()] = l.GetLayerWeights()
		}
	}

	config := pythonConfig{
		ModelConfig:            modelConfig,
		SaveDir:                filepath.Join(tempDir, tempModelDir),
		ModelDefinitionSaveDir: modelDefinitionSaveDir,
		Loss:                   string(loss),
		Optimizer:              optimizer.GetKerasLayerConfig(),
		Weights:                weights,
	}

	configBytes, e := json.Marshal(config)
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}

	ignoreRegex := regexp.MustCompile("# tfkg-ignore.*# tfkg-ignore-end")

	layerTypesDefined := make(map[string]bool)
	var customDefinitions []string
	for _, l := range m.layers {
		if len(l.GetCustomLayerDefinition()) == 0 {
			continue
		}
		definition := l.GetCustomLayerDefinition()
		definition = ignoreRegex.ReplaceAllString(definition, "")
		if _, ok := layerTypesDefined[definition]; !ok {
			customDefinitions = append(customDefinitions, definition)
			layerTypesDefined[definition] = true
		}
	}

	tempPythonPath := filepath.Join(tempDir, "tfkg_create_model.py")

	e = ioutil.WriteFile(tempPythonPath, []byte(GetPythonCode(customDefinitions)), os.ModePerm)
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}
	defer os.Remove(tempPythonPath)

	tempConfigPath := filepath.Join(tempDir, "tfkg_config.json")

	e = ioutil.WriteFile(tempConfigPath, configBytes, os.ModePerm)
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}
	defer os.Remove(tempPythonPath)

	cmd := exec.Command("python", tempPythonPath, tempConfigPath)
	output, e := cmd.CombinedOutput()
	if e != nil {
		m.errorHandler.Error(e)
	}
	fmt.Println(string(output))
	if e != nil {
		return e
	}

	m.model, e = tf.LoadSavedModel(filepath.Join(tempDir, tempModelDir), []string{"serve"}, nil)
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}

	m.pbCache, e = ioutil.ReadFile(filepath.Join(tempDir, tempModelDir, "saved_model.pb"))
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}

	e = os.RemoveAll(filepath.Join(tempDir, tempModelDir))
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}

	return nil
}

type kerasModelConfigStruct struct {
	ClassName string `json:"class_name"`
	Config    struct {
		Name         string          `json:"name"`
		Layers       []interface{}   `json:"layers"`
		InputLayers  [][]interface{} `json:"input_layers"`
		OutputLayers [][]interface{} `json:"output_layers"`
	} `json:"config"`
	KerasVersion string `json:"keras_version"`
	Backend      string `json:"backend"`
}

func (m *TfkgModel) generateKerasDefinitionJson() (string, error) {
	var inputLayerConfigs [][]interface{}
	var outputLayerConfigs [][]interface{}
	var layerConfigs []interface{}
	for i, l := range m.layers {
		if _, ok := l.(*layer.LInput); ok {
			inputLayerConfigs = append(inputLayerConfigs, []interface{}{
				l.GetName(),
				0,
				0,
			})
		}
		if i+1 == len(m.layers) {
			outputLayerConfigs = append(outputLayerConfigs, []interface{}{
				l.GetName(),
				0,
				0,
			})
		}
		layerConfigs = append(layerConfigs, l.GetKerasLayerConfig())
	}
	config := kerasModelConfigStruct{
		ClassName: "Functional",
		Config: struct {
			Name         string          `json:"name"`
			Layers       []interface{}   `json:"layers"`
			InputLayers  [][]interface{} `json:"input_layers"`
			OutputLayers [][]interface{} `json:"output_layers"`
		}{
			Name:         "model",
			Layers:       layerConfigs,
			InputLayers:  inputLayerConfigs,
			OutputLayers: outputLayerConfigs,
		},
		KerasVersion: "2.6.0",
		Backend:      "tensorflow",
	}

	jsonBytes, e := json.Marshal(config)
	if e != nil {
		m.errorHandler.Error(e)
		return "", e
	}

	return string(jsonBytes), nil
}

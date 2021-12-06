package model

import (
	"fmt"
	"github.com/codingbeard/cberrors"
	"github.com/codingbeard/cblog"
	"github.com/codingbeard/tfkg/callback"
	"github.com/codingbeard/tfkg/data"
	"github.com/codingbeard/tfkg/layer"
	"github.com/codingbeard/tfkg/metric"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

var (
	tempModelDir = "tfkg_temp_model"
)

type TfModel struct {
	model        *tf.SavedModel
	baseModelDir string
	layers       []layer.Layer
	isSequential bool
	pbCache      []byte

	errorHandler *cberrors.ErrorsContainer
	logger       *cblog.Logger
}

func NewSequentialModel(
	logger *cblog.Logger,
	errorHandler *cberrors.ErrorsContainer,
	layers ...layer.Layer,
) *TfModel {
	return &TfModel{
		isSequential: true,
		layers:       layers,
		baseModelDir: tempModelDir,
		errorHandler: errorHandler,
		logger:       logger,
	}
}

func LoadModel(
	errorHandler *cberrors.ErrorsContainer,
	logger *cblog.Logger,
	dir string,
) (*TfModel, error) {
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

	return &TfModel{
		model:        m,
		baseModelDir: dir,
		pbCache:      pbCache,
		errorHandler: errorHandler,
		logger:       logger,
	}, nil
}

func (m *TfModel) CompileAndLoad(batchSize int, pythonPath string, createModelFunc ...func(tempDir, tempPythonGenerationFile string) error) error {
	// TODO: this is nasty, replace it with json configs
	tempDir := filepath.Join(os.TempDir(), "/tfkg")
	code, e := m.generatePythonModelCode(tempDir, batchSize)
	if e != nil {
		return e
	}

	e = os.MkdirAll(tempDir, os.ModePerm)
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}

	generatorFileName := "tfkg_temp_model_generator.py"
	e = ioutil.WriteFile(filepath.Join(tempDir, generatorFileName), []byte(code), os.ModePerm)
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}

	if len(createModelFunc) == 1 {
		e = createModelFunc[0](tempDir, generatorFileName)
		if e != nil {
			return e
		}
	} else {
		e = func(tempDir string, tempPythonGenerationFile string) error {
			cmd := exec.Command(pythonPath, tempPythonGenerationFile)
			cmd.Dir = tempDir
			output, e := cmd.CombinedOutput()
			if e != nil {
				m.errorHandler.Error(e)
			}
			fmt.Println(string(output))
			return e
		}(tempDir, generatorFileName)
		if e != nil {
			return e
		}
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

	e = os.Remove(filepath.Join(tempDir, generatorFileName))
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}

	return nil
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

func (m *TfModel) Fit(
	trainSignature string,
	valSignature string,
	dataset data.Dataset,
	config FitConfig,
) {
	trainSig := m.model.Signatures[trainSignature]
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

	for _, met := range config.Metrics {
		met.Init()
	}
	for _, call := range config.Callbacks {
		e := call.Init()
		if e != nil {
			m.errorHandler.Error(e)
			return
		}
	}

	for epoch := 1; epoch <= config.Epochs; epoch++ {

		generatorChan := dataset.
			SetMode(data.GeneratorModeTrain).
			GeneratorChan(config.BatchSize, config.PreFetch)

		labelOp := m.model.Graph.Operation(fmt.Sprintf("%s_%s", trainSignature, "y")).Output(0)
		// TODO: change class weights to a single tensor
		posWeightOp := m.model.Graph.Operation(fmt.Sprintf("%s_%s", trainSignature, "pos_weight")).Output(0)
		negWeightOp := m.model.Graph.Operation(fmt.Sprintf("%s_%s", trainSignature, "neg_weight")).Output(0)

		var inputOps []tf.Output
		for _, columnName := range dataset.GetColumnNames() {
			inputOps = append(inputOps, m.model.Graph.Operation(fmt.Sprintf("%s_%s", trainSignature, columnName)).Output(0))
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

			x, y, posWeight, negWeight := generatorBatch.X, generatorBatch.Y, generatorBatch.PosWeight, generatorBatch.NegWeight

			inputs := map[tf.Output]*tf.Tensor{
				labelOp: y,
				// TODO: change class weights to a single tensor
				posWeightOp: posWeight,
				negWeightOp: negWeight,
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
		for _, met := range config.Metrics {
			met.Reset()
		}
		if config.Validation {
			valSig := m.model.Signatures[valSignature]
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

			valLabelOp := m.model.Graph.Operation(fmt.Sprintf("%s_%s", valSignature, "y")).Output(0)
			// TODO: change class weights to a single tensor
			valPosWeightOp := m.model.Graph.Operation(fmt.Sprintf("%s_%s", valSignature, "pos_weight")).Output(0)
			valNegWeightOp := m.model.Graph.Operation(fmt.Sprintf("%s_%s", valSignature, "neg_weight")).Output(0)

			var valInputOps []tf.Output
			for _, columnName := range dataset.GetColumnNames() {
				valInputOps = append(valInputOps, m.model.Graph.Operation(fmt.Sprintf("%s_%s", valSignature, columnName)).Output(0))
			}

			batch = 1
			totalBatches := dataset.Len() / config.BatchSize
			valTotalLoss := float64(0)
			for generatorBatch := range generatorChan {
				if halt {
					break
				}

				x, y, posWeight, negWeight := generatorBatch.X, generatorBatch.Y, generatorBatch.PosWeight, generatorBatch.NegWeight

				inputs := map[tf.Output]*tf.Tensor{
					valLabelOp: y,
					// TODO: change class weights to a single tensor
					valPosWeightOp: posWeight,
					valNegWeightOp: negWeight,
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

func (m *TfModel) Evaluate(
	evaluateSignature string,
	mode data.GeneratorMode,
	dataset data.Dataset,
	config EvaluateConfig,
) {

	for _, met := range config.Metrics {
		met.Init()
	}
	for _, call := range config.Callbacks {
		e := call.Init()
		if e != nil {
			m.errorHandler.Error(e)
			return
		}
	}
	evaluateSig := m.model.Signatures[evaluateSignature]
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

	evaluateLabelOp := m.model.Graph.Operation(fmt.Sprintf("%s_%s", evaluateSignature, "y")).Output(0)
	// TODO: change class weights to a single tensor
	evaluatePosWeightOp := m.model.Graph.Operation(fmt.Sprintf("%s_%s", evaluateSignature, "pos_weight")).Output(0)
	evaluateNegWeightOp := m.model.Graph.Operation(fmt.Sprintf("%s_%s", evaluateSignature, "neg_weight")).Output(0)

	var evaluateInputOps []tf.Output
	for _, columnName := range dataset.GetColumnNames() {
		evaluateInputOps = append(evaluateInputOps, m.model.Graph.Operation(fmt.Sprintf("%s_%s", evaluateSignature, columnName)).Output(0))
	}

	batch := 1
	totalBatches := dataset.Len() / config.BatchSize
	evaluateTotalLoss := float64(0)
	var halt bool
	for generatorBatch := range generatorChan {
		if halt {
			break
		}

		x, y, posWeight, negWeight := generatorBatch.X, generatorBatch.Y, generatorBatch.PosWeight, generatorBatch.NegWeight

		inputs := map[tf.Output]*tf.Tensor{
			evaluateLabelOp: y,
			// TODO: change class weights to a single tensor
			evaluatePosWeightOp: posWeight,
			evaluateNegWeightOp: negWeight,
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

func (m *TfModel) getMetricLogs(
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

func (m *TfModel) processCallbacks(
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

func (m *TfModel) callCallbacks(
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

func (m *TfModel) Save(dir string) error {
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

func (m *TfModel) generatePythonModelCode(tempDir string, batchSize int) (string, error) {
	// TODO: this is nasty, replace it with json configs
	imports := make(map[string]bool)

	var inputVariableNames []string
	var inputShapes []string
	var inputDtypes []string
	var inputBlanks []string
	var outputVariableName string
	var labelSpec string
	var outputBlank string

	var layerDefs []string
	var lastLayer layer.Layer
	for i, l := range m.layers {
		if _, ok := l.(*layer.Input); ok {
			inputVariableNames = append(inputVariableNames, l.GetPythonVariableName())
			inputShapes = append(inputShapes, strings.ReplaceAll(l.GetShape().String(), "?", "None"))
			inputDtypes = append(inputDtypes, string(l.GetDtype()))
			inputBlanks = append(inputBlanks, fmt.Sprintf(
				"%s = tf.zeros(shape=[%d, %d], dtype=%s)",
				inputVariableNames[i],
				batchSize,
				l.GetShape().Size(1),
				inputDtypes[i],
			))

		} else {
			if m.isSequential {
				l.SetInput([]layer.Layer{lastLayer})
			} else {
				panic("functional model code generation not supported yet")
			}
		}
		if i+1 == len(m.layers) {
			outputVariableName = l.GetPythonVariableName()
			labelSpec = fmt.Sprintf(
				"tf.TensorSpec(shape=(None, 1), dtype=tf.int32)",
			)
			outputBlank = fmt.Sprintf(
				"y = tf.zeros(shape=[%d, 1], dtype=tf.int32)",
				batchSize,
			)
		}
		imports[l.GetImport()] = true
		layerDefLines := strings.Split(fmt.Sprintf(
			"%s = %s",
			l.GetPythonVariableName(),
			l.GetPythonDefinitionString(),
		), "\n")

		for lineOffset := range layerDefLines {
			layerDefLines[lineOffset] = "        " + layerDefLines[lineOffset]
		}

		layerDefs = append(layerDefs, strings.Join(layerDefLines, "\n"))
		lastLayer = l
	}

	var importsSlice []string
	for imp := range imports {
		importsSlice = append(importsSlice, imp)
	}

	var inputSpecs []string
	for i := range inputShapes {
		inputSpecs = append(inputSpecs, fmt.Sprintf("tf.TensorSpec(shape=%s, dtype=%s)", inputShapes[i], inputDtypes[i]))
	}

	return fmt.Sprintf(
		`
import os
import logging
import datetime
import tensorflow as tf
from tensorflow.keras import Model
%s

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.disable(logging.WARNING)


def timestamp(message):
    print(
        datetime.datetime.now().strftime("%s") +
        " " +
        message
    )


class GolangModel(tf.Module):
    def __init__(self):
        super().__init__()

        self.batch_size = %d

%s

        self._model = Model([%s], %s)
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self._optimizer = tf.keras.optimizers.Adam()
        loss_func = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction="none"
        )

        def loss(y_true, y_pred, pos_weight, neg_weight):
            weighted_loss = tf.multiply(
                loss_func(y_true, y_pred),
                tf.where(
                    tf.greater_equal(
                        tf.reshape(y, shape=[self.batch_size]),
                        tf.ones(dtype=tf.int32, shape=[self.batch_size])
                    ),
                    tf.repeat(pos_weight, self.batch_size),
                    tf.repeat(neg_weight, self.batch_size),
                )
            )
            return tf.reduce_mean(weighted_loss)

        self._loss = loss

    @tf.function(input_signature=[
        %s
        %s,
        tf.TensorSpec(shape=1, dtype=tf.float32),
        tf.TensorSpec(shape=1, dtype=tf.float32),
    ])
    def learn(
            self,
            %s
            y,
            neg_weight,
            pos_weight,
    ):
        self._global_step.assign_add(1)
        with tf.GradientTape() as tape:
            logits = self._model([%s], training=True)
            loss = self._loss(y, logits, pos_weight, neg_weight)

        gradient = tape.gradient(
            loss,
            self._model.trainable_variables
        )
        self._optimizer.apply_gradients(
            zip(gradient, self._model.trainable_variables)
        )
        return [
            loss,
            logits
        ]

    @tf.function(input_signature=[
        %s
        %s,
        tf.TensorSpec(shape=1, dtype=tf.float32),
        tf.TensorSpec(shape=1, dtype=tf.float32),
    ])
    def evaluate(
            self,
            %s
            y,
            neg_weight,
            pos_weight,
    ):
        self._global_step.assign_add(1)
        with tf.GradientTape() as tape:
            logits = self._model([%s], training=True)
            loss = self._loss(y, logits, pos_weight, neg_weight)

        return [
            loss,
            logits
        ]

    @tf.function(input_signature=[
        %s
    ])
    def predict(
            self,
            %s
    ):
        return [self._model(%s)]


timestamp("Initialising model")

gm = GolangModel()

%s
%s

timestamp("Tracing learn")

gm.learn(
    %s
    y,
    tf.convert_to_tensor([1], dtype=tf.float32),
    tf.convert_to_tensor([1], dtype=tf.float32),
)

timestamp("Tracing evaluate")

gm.evaluate(
    %s
    y,
    tf.convert_to_tensor([1], dtype=tf.float32),
    tf.convert_to_tensor([1], dtype=tf.float32),
)

timestamp("Tracing predict")

gm.predict(%s)

timestamp("Saving model")

tf.saved_model.save(
    gm,
    "%s",
    signatures={
        "learn": gm.learn,
        "evaluate": gm.evaluate,
        "predict": gm.predict,
    },
)

timestamp("Completed model base")
`,
		strings.Join(importsSlice, "\n"),
		"%d/%m/%Y %H:%M:%S",
		batchSize,
		strings.Join(layerDefs, "\n"),
		strings.Join(inputVariableNames, ", "),
		outputVariableName,
		strings.Join(inputSpecs, "\n        ")+",",
		labelSpec,
		strings.Join(inputVariableNames, ",\n            ")+",",
		strings.Join(inputVariableNames, ", "),
		strings.Join(inputSpecs, "\n        ")+",",
		labelSpec,
		strings.Join(inputVariableNames, ",\n            ")+",",
		strings.Join(inputVariableNames, ", "),
		strings.Join(inputSpecs, "\n        ")+",",
		strings.Join(inputVariableNames, ",\n            ")+",",
		strings.Join(inputVariableNames, ", "),
		strings.Join(inputBlanks, "\n"),
		outputBlank,
		strings.Join(inputVariableNames, ",\n    ")+",",
		strings.Join(inputVariableNames, ",\n    ")+",",
		strings.Join(inputVariableNames, ","),
		filepath.Join(tempDir, tempModelDir),
	), nil
}

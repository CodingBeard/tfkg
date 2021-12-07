package model

import (
	"encoding/json"
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

type TfkgModel struct {
	model        *tf.SavedModel
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
) *TfkgModel {
	return &TfkgModel{
		isSequential: true,
		layers:       layers,
		errorHandler: errorHandler,
		logger:       logger,
	}
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
		model:        m,
		pbCache:      pbCache,
		errorHandler: errorHandler,
		logger:       logger,
	}, nil
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

		labelOp := m.model.Graph.Operation(fmt.Sprintf("learn_%s", "y")).Output(0)
		// TODO: change class weights to a single tensor
		posWeightOp := m.model.Graph.Operation(fmt.Sprintf("learn_%s", "pos_weight")).Output(0)
		negWeightOp := m.model.Graph.Operation(fmt.Sprintf("learn_%s", "neg_weight")).Output(0)

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
			// TODO: change class weights to a single tensor
			valPosWeightOp := m.model.Graph.Operation(fmt.Sprintf("evaluate_%s", "pos_weight")).Output(0)
			valNegWeightOp := m.model.Graph.Operation(fmt.Sprintf("evaluate_%s", "neg_weight")).Output(0)

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

func (m *TfkgModel) Evaluate(
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
	// TODO: change class weights to a single tensor
	evaluatePosWeightOp := m.model.Graph.Operation(fmt.Sprintf("evaluate_%s", "pos_weight")).Output(0)
	evaluateNegWeightOp := m.model.Graph.Operation(fmt.Sprintf("evaluate_%s", "neg_weight")).Output(0)

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
	BatchSize   int    `json:"batch_size"`
	ModelConfig string `json:"model_config"`
	SaveDir     string `json:"save_dir"`
}

func (m *TfkgModel) CompileAndLoad(batchSize int) error {
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

	config := pythonConfig{
		BatchSize:   batchSize,
		ModelConfig: modelConfig,
		SaveDir:     filepath.Join(tempDir, tempModelDir),
	}

	configBytes, e := json.Marshal(config)
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}

	cmd := exec.Command("python", "-c", m.getPythonModelCode())
	stdinPipe, e := cmd.StdinPipe()
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}
	_, e = stdinPipe.Write(configBytes)
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}
	e = stdinPipe.Close()
	if e != nil {
		m.errorHandler.Error(e)
		return e
	}
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
	var lastLayer layer.Layer
	for i, l := range m.layers {
		if _, ok := l.(*layer.Input); ok {
			inputLayerConfigs = append(inputLayerConfigs, []interface{}{
				l.GetName(),
				0,
				0,
			})
		} else {
			if m.isSequential {
				l.SetInput([]layer.Layer{lastLayer})
			} else {
				panic("functional model code generation not supported yet")
			}
		}
		if i+1 == len(m.layers) {
			outputLayerConfigs = append(outputLayerConfigs, []interface{}{
				l.GetName(),
				0,
				0,
			})
		}
		layerConfigs = append(layerConfigs, l.GetKerasLayerConfig())
		lastLayer = l
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

func (m *TfkgModel) getPythonModelCode() string {
	// TODO somehow move this into a separate python file and load it into a resource at build time
	return `import json
import os
import logging
import sys

import tensorflow as tf
from tensorflow.keras import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.disable(logging.WARNING)

config = json.load(sys.stdin)

model = tf.keras.models.model_from_json(config["model_config"])
learn_input_signature = [
    tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
    tf.TensorSpec(shape=1, dtype=tf.float32),
    tf.TensorSpec(shape=1, dtype=tf.float32),
]
predict_input_signature = []

zero_inputs = []

for model_layer in model.layers:
    if type(model_layer) == tf.keras.layers.InputLayer:
        input_shape = [config["batch_size"]]
        for dim in model_layer.input_shape[0][3:]:
            input_shape.append(dim)
        zero_inputs.append(
            tf.zeros(shape=input_shape, dtype=model_layer.dtype)
        )
        learn_input_signature.append(tf.TensorSpec(
            shape=model_layer.input_shape[0][2:],
            dtype=model_layer.dtype,
        ))
        predict_input_signature.append(tf.TensorSpec(
            shape=model_layer.input_shape[0][2:],
            dtype=model_layer.dtype,
        ))

evaluate_input_signature = learn_input_signature


class GolangModel(tf.Module):
    def __init__(self):
        super().__init__()

        self.batch_size = config["batch_size"]
        self._model = model

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
                        tf.reshape(y_true, shape=[self.batch_size]),
                        tf.ones(dtype=tf.int32, shape=[self.batch_size])
                    ),
                    tf.repeat(pos_weight, self.batch_size),
                    tf.repeat(neg_weight, self.batch_size),
                )
            )
            return tf.reduce_mean(weighted_loss)

        self._loss = loss

    @tf.function(input_signature=learn_input_signature)
    def learn(
            self,
            y,
            neg_weight,
            pos_weight,
            *inputs
    ):
        self._global_step.assign_add(1)
        with tf.GradientTape() as tape:
            logits = self._model(list(inputs), training=True)
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

    @tf.function(input_signature=evaluate_input_signature)
    def evaluate(
            self,
            y,
            neg_weight,
            pos_weight,
            *inputs
    ):
        self._global_step.assign_add(1)
        with tf.GradientTape() as tape:
            logits = self._model(list(inputs), training=True)
            loss = self._loss(y, logits, pos_weight, neg_weight)

        return [
            loss,
            logits
        ]

    @tf.function(input_signature=predict_input_signature)
    def predict(
            self,
            *inputs,
    ):
        return [self._model(list(inputs))]


print("Initialising model")

gm = GolangModel()

y_zeros = tf.zeros(shape=[config["batch_size"], 1], dtype=tf.int32)

print("Tracing learn")

gm.learn(
    y_zeros,
    tf.convert_to_tensor([1], dtype=tf.float32),
    tf.convert_to_tensor([1], dtype=tf.float32),
    *zero_inputs,
)

print("Tracing evaluate")

gm.evaluate(
    y_zeros,
    tf.convert_to_tensor([1], dtype=tf.float32),
    tf.convert_to_tensor([1], dtype=tf.float32),
    *zero_inputs,
)

print("Tracing predict")

gm.predict(*zero_inputs)

print("Saving model")

tf.saved_model.save(
    gm,
    config["save_dir"],
    signatures={
        "learn": gm.learn,
        "evaluate": gm.evaluate,
        "predict": gm.predict,
    },
)

print("Completed model base")
`
}

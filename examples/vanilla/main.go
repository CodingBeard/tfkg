package main

import (
	"fmt"
	"github.com/codingbeard/cberrors"
	"github.com/codingbeard/cberrors/iowriterprovider"
	"github.com/codingbeard/cblog"
	"github.com/codingbeard/tfkg/callback"
	"github.com/codingbeard/tfkg/data"
	"github.com/codingbeard/tfkg/metric"
	"github.com/codingbeard/tfkg/model"
	"github.com/codingbeard/tfkg/optimizer"
	"github.com/codingbeard/tfkg/preprocessor"
	"os"
	"path/filepath"
	"time"
)

func main() {
	// This is where the trained model will be saved
	saveDir := filepath.Join("../../logs", fmt.Sprintf("transfer-learning-%d", time.Now().Unix()))
	e := os.MkdirAll(saveDir, os.ModePerm)
	if e != nil {
		panic(e)
	}
	// This is where the model with transferred weights will be saved
	transferredSaveDir := filepath.Join(saveDir, "transferred")
	e = os.MkdirAll(transferredSaveDir, os.ModePerm)
	if e != nil {
		panic(e)
	}

	// Create a logger pointed at the save dir
	logger, e := cblog.NewLogger(cblog.LoggerConfig{
		LogLevel:           cblog.DebugLevel,
		Format:             "%{time:2006-01-02 15:04:05.000} : %{file}:%{line} : %{message}",
		LogToFile:          true,
		FilePath:           filepath.Join(saveDir, "training.log"),
		FilePerm:           os.ModePerm,
		LogToStdOut:        true,
		SetAsDefaultLogger: true,
	})
	if e != nil {
		panic(e)
	}

	// Error handler with stack traces
	errorHandler := cberrors.NewErrorContainer(iowriterprovider.New(logger))

	// Where the cached tokenizers and divisors will go, if you change your data you'll need to clear this
	cacheDir := "training-cache"

	// Create a dataset for evaluation. iris.data is in the format: float32, float32, float32, float32, className
	// This means our categoryOffset is 4. The dataset will automatically pass this value in as the label Tensor when training and evaluating
	// We allocate 100% of the data to testing (TestPercent: 1)
	// We define a data processor for the four float32 data points. The name will be used for the tokenizer or divisor cache file
	// The lineOffset is 0 because the data is the first thing in the csv row and the dataLength is 4 because there are 4 floats to train
	// We use a preprocessor.ReadCsvFloat32s because under the hood a lineOffset: 0 dataLength: 4 will grab the first four elements of the csv row and return them as a csv string. It will convert the string to a slice of float32 values
	// We use a preprocessor.ConvertDivisorToFloat32SliceTensor to convert that slice of floats into a tensorflow Tensor. The output of this function will be passed to the model for training and evaluating
	dataset, e := data.NewSingleFileDataset(
		logger,
		errorHandler,
		data.SingleFileDatasetConfig{
			FilePath:          "data/iris.data",
			CacheDir:          cacheDir,
			TestPercent:       1,
			IgnoreParseErrors: true,
		},
		preprocessor.NewSparseCategoricalTokenizingYProcessor(
			errorHandler,
			cacheDir,
			4,
		),
		preprocessor.NewProcessor(
			errorHandler,
			"petal_sizes",
			preprocessor.ProcessorConfig{
				CacheDir:   cacheDir,
				LineOffset: 0,
				DataLength: 4,
				Reader:     preprocessor.ReadCsvFloat32s,
				Converter:  preprocessor.ConvertDivisorToFloat32SliceTensor,
			},
		),
	)
	if e != nil {
		errorHandler.Error(e)
		return
	}

	logger.InfoF("main", "Shuffling dataset")
	// This will shuffle the data in a deterministic fashion, change 1 to time.Now().UnixNano() for a different shuffle each training session
	dataset.Shuffle(1)

	logger.InfoF("main", "Loading model")
	// Load a vanilla Tensorflow/Keras model saved by generate_vanilla_model.py
	m, e := model.LoadVanillaModel(
		errorHandler,
		logger,
		"vanilla_model",
		model.LossSparseCategoricalCrossentropy,
		optimizer.Adam(),
	)

	logger.InfoF("main", "Evaluating model")
	// Evaluate the loaded vanilla model, we should see a high accuracy in the logs
	m.Evaluate(
		data.GeneratorModeTest,
		dataset,
		model.EvaluateConfig{
			BatchSize: 3,
			PreFetch:  10,
			Verbose:   1,
			Metrics: []metric.Metric{
				&metric.SparseCategoricalAccuracy{
					Name:       "acc",
					Confidence: 0.5,
					Average:    true,
				},
			},
			Callbacks: []callback.Callback{
				&callback.Logger{
					FileLogger: logger,
				},
			},
		},
	)
	fmt.Println()

	/*
		Example output:
			50/50 [==============================] - 1s 4ms/step - loss: 0.1421 - accuracy: 0.9733
			2021-12-29 20:59:29.923 : log.go:147 : Logger initialised
			2021-12-29 20:59:29.928 : single_file_dataset.go:87 : Initialising single file dataset at: data/iris.data
			2021-12-29 20:59:29.941 : single_file_dataset.go:204 : Loading line offsets and stats from cache file
			2021-12-29 20:59:29.945 : single_file_dataset.go:213 : Found 150 rows. Got class counts: map[int]int{0:50, 1:50, 2:50} Got class weights: map[int]float32{0:1, 1:1, 2:1}
			2021-12-29 20:59:29.949 : single_file_dataset.go:385 : Loaded All Pre-Processors
			2021-12-29 20:59:29.952 : main.go:97 : Shuffling dataset
			2021-12-29 20:59:29.953 : main.go:101 : Loading model
			2021-12-29 20:59:29.955 : model.go:163 : Loading vanilla model. If anything goes wrong python error messages will be printed out.
			Loading Vanilla model
			Tracing learn
			Tracing evaluate
			Tracing predict
			Tracing get_weights
			Saving model
			Completed model base
		    2021-12-29 20:59:47.673 : main.go:111 : Evaluating model
			2021-12-29 20:59:48.101 : logger.go:110 : End 1 50/50 (0s/0s) test_loss: 0.1438 test_acc: 0.9728
	*/
}

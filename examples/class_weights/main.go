package main

import (
	"fmt"
	"github.com/codingbeard/cberrors"
	"github.com/codingbeard/cberrors/iowriterprovider"
	"github.com/codingbeard/cblog"
	"github.com/codingbeard/tfkg/callback"
	"github.com/codingbeard/tfkg/data"
	"github.com/codingbeard/tfkg/layer"
	"github.com/codingbeard/tfkg/metric"
	"github.com/codingbeard/tfkg/model"
	"github.com/codingbeard/tfkg/optimizer"
	"github.com/codingbeard/tfkg/preprocessor"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"math/rand"
	"os"
	"path/filepath"
	"time"
)

func main() {
	logsDir := filepath.Join("../../logs", fmt.Sprintf("class-weights-%d", time.Now().Unix()))
	e := os.MkdirAll(logsDir, os.ModePerm)

	if e != nil {
		panic(e)
	}
	// Create a logger pointed at the save dir
	logger, e := cblog.NewLogger(cblog.LoggerConfig{
		LogLevel:           cblog.DebugLevel,
		Format:             "%{time:2006-01-02 15:04:05.000} : %{file}:%{line} : %{message}",
		LogToFile:          true,
		FilePath:           filepath.Join(logsDir, "training.log"),
		FilePerm:           os.ModePerm,
		LogToStdOut:        true,
		SetAsDefaultLogger: true,
	})
	if e != nil {
		panic(e)
	}

	// Error handler with stack traces
	errorHandler := cberrors.NewErrorContainer(iowriterprovider.New(logger))

	// Create a new Values dataset, we can pass in values without having to read from a CSV file
	dataset, e := data.NewValuesDataset(
		logger,
		errorHandler,
		data.ValuesDatasetConfig{
			TrainPercent: 0.8,
			ValPercent:   0.1,
			TestPercent:  0.1,
		},
		preprocessor.NewProcessor(
			errorHandler,
			"y",
			preprocessor.ProcessorConfig{
				Converter: preprocessor.ConvertInterfaceToInt32SliceTensor,
			},
		),
		preprocessor.NewProcessor(
			errorHandler,
			"floats",
			preprocessor.ProcessorConfig{
				Converter: preprocessor.ConvertInterfaceFloat32SliceToTensor,
			},
		),
	)
	if e != nil {
		errorHandler.Error(e)
		return
	}

	var y []interface{}
	var x []interface{}
	// generate random input data with imbalanced classes, if class_weighting works the accuracy should be 0
	for i := 0; i < 100000; i++ {
		y = append(y, int32(0))
		y = append(y, int32(1))
		y = append(y, int32(1))
		y = append(y, int32(2))
		y = append(y, int32(2))
		y = append(y, int32(2))
		for j := 0; j < 6; j++ {
			x = append(x, []float32{
				rand.Float32(),
				rand.Float32(),
				rand.Float32(),
				rand.Float32(),
			})
		}
	}

	e = dataset.SetValues(y, x)
	if e != nil {
		errorHandler.Error(e)
		return
	}

	logger.InfoF("main", "Shuffling dataset")
	// This will shuffle the data in a deterministic fashion, change 1 to time.Now().UnixNano() for a different shuffle each training session
	dataset.Shuffle(1)

	// Define a simple keras style Sequential model with two hidden Dense layers
	m := model.NewSequentialModel(
		logger,
		errorHandler,
		layer.Input().SetInputShape(tf.MakeShape(-1, 4)).SetDtype(layer.Float32),
		layer.Dense(100).SetActivation("swish"),
		layer.Dense(100).SetActivation("swish"),
		layer.Dense(float64(dataset.NumCategoricalClasses())).SetActivation("softmax"),
	)

	// This part is pretty nasty under the hood. Effectively it will generate some python code for our model and execute it to save the model in a format we can load and train
	// A python binary must be available to use for this to work
	e = m.CompileAndLoad(model.LossSparseCategoricalCrossentropy, optimizer.Adam(), logsDir)
	if e != nil {
		return
	}

	logger.InfoF("main", "Training model")

	// Train the model.
	// Most of this should look familiar to anyone who has used tensorflow/keras
	// The key points are:
	//      We pass the data through 10 times (Epochs: 10)
	//      We enable validation, which will evaluate the model on the validation portion of the dataset above (Validation: true)
	//      We continuously (and concurrently) pre-fetch 10 batches to speed up training, though with 150 samples this has almost no effect
	// 		We calculate the accuracy of the model on training and validation datasets (metric.SparseCategoricalAccuracy)
	//		We log the training results to stdout (Verbose:1, callback.Logger)
	m.Fit(
		dataset,
		model.FitConfig{
			Epochs:     10,
			Validation: true,
			BatchSize:  1000,
			PreFetch:   10,
			Verbose:    1,
			Metrics: []metric.Metric{
				&metric.SparseCategoricalAccuracy{
					Name:       "acc",
					Confidence: 0.5,
					Average:    true,
				},
			},
			Callbacks: []callback.Callback{
				&callback.Logger{
					FileLogger:     logger,
					Progress:       true,
					ProgressLogDir: logsDir,
				},
			},
		},
	)

	logger.InfoF("main", "Finished training")

}

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
	"os"
	"path/filepath"
	"time"
)

func main() {
	// This is where the trained model will be saved
	saveDir := filepath.Join("../../logs", fmt.Sprintf("bench-%d", time.Now().Unix()))
	e := os.MkdirAll(saveDir, os.ModePerm)
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

	// Create a new Values dataset, we can pass in values without having to read from a CSV file
	dataset, e := data.NewValuesDataset(
		logger,
		errorHandler,
		data.ValuesDatasetConfig{
			TrainPercent: 1,
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
	for i := 0; i < 29000; i++ {
		var xRow []float32
		for j := 0; j < 1000; j++ {
			xRow = append(xRow, float32(j))
		}
		x = append(x, xRow)
		if i%2 == 0 {
			y = append(y, int32(0))
		} else {
			y = append(y, int32(1))
		}
	}

	e = dataset.SetValues(y, x)
	if e != nil {
		errorHandler.Error(e)
		return
	}

	logger.InfoF("main", "Loading model")
	m := model.NewSequentialModel(
		logger,
		errorHandler,
		layer.Input().SetInputShape(tf.MakeShape(-1, 1000)).SetDtype(layer.Float32),
		layer.Embedding(1000, 32).SetBatchInputShape([]interface{}{nil, 1000}).SetInputLength(1000),
		layer.CuDNNLSTM(128),
		layer.Dense(1024).SetActivation("swish"),
		layer.Dense(1).SetActivation("sigmoid"),
	)

	e = m.CompileAndLoad(model.CompileConfig{
		Loss:             model.LossBinaryCrossentropy,
		Optimizer:        optimizer.Adam(),
		ModelInfoSaveDir: saveDir,
		BatchSize:        500,
	})
	if e != nil {
		return
	}

	start := time.Now().Unix()
	m.Fit(
		dataset,
		model.FitConfig{
			Epochs:    1,
			BatchSize: 500,
			PreFetch:  10,
			Metrics: []metric.Metric{
				&metric.BinaryAccuracy{
					Name:       "acc",
					Confidence: 0.5,
					Average:    true,
				},
				&metric.BinaryFprAtTpr{
					Name:     "specAtSen99",
					Tpr:      0.99,
					Attempts: 100,
				},
			},
			Callbacks: []callback.Callback{
				&callback.Logger{
					FileLogger: logger,
				},
			},
			Verbose: 0,
		},
	)
	end := time.Now().Unix()
	fmt.Println()
	fmt.Println(end-start, "s - Golang")
}

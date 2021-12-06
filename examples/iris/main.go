package main

import (
	"github.com/codingbeard/cberrors"
	"github.com/codingbeard/cberrors/iowriterprovider"
	"github.com/codingbeard/cblog"
	"github.com/codingbeard/tfkg/callback"
	"github.com/codingbeard/tfkg/data"
	"github.com/codingbeard/tfkg/layer"
	"github.com/codingbeard/tfkg/metric"
	"github.com/codingbeard/tfkg/model"
	"github.com/codingbeard/tfkg/preprocessor"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"os"
	"path/filepath"
)

func main() {
	// This is where the trained model will be saved
	saveDir := "examples/iris/saved_models/trained_model"
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

	// Define a simple keras style Sequential model with two hidden Dense layers
	// Note that the input name "petal_sizes" MUST correspond to the name used in the dataset further down
	m := model.NewSequentialModel(
		logger,
		errorHandler,
		layer.NewInput(tf.MakeShape(-1, 4), layer.Float32, layer.InputConfig{Name: "petal_sizes"}),
		layer.NewDense(100, layer.Float32, layer.DenseConfig{Activation: "swish"}),
		layer.NewDense(100, layer.Float32, layer.DenseConfig{Activation: "swish"}),
		layer.NewDense(3, layer.Float32, layer.DenseConfig{Activation: "softmax"}),
	)

	// This part is pretty nasty under the hood. Effectively it will generate some python code for our model and then save the model in a format we can load and train
	// The pythonPath must be a python binary which knows about the corresponding tensorflow libraries. In this example it is the python path in the docker container provided
	e = m.CompileAndLoad(
		3,
		"/usr/local/bin/python",
	)
	if e != nil {
		return
	}

	// Where the cached tokenizers and divisors will go, if you change your data you'll need to clear this
	cacheDir := "examples/iris/training-cache"

	// Create a dataset for training and evaluation. iris.data is in the format: float32, float32, float32, float32, className
	// This means our categoryOffset is 4. The dataset will automatically pass this value in as the label Tensor when training and evaluating
	// If the category is not an int, a tokenizer will be created to automatically convert string categories to ints in a sparse categorical format
	// We allocate 80% of the data to training (TrainPercent: 0.8)
	// We allocate 10% of the data to validation (ValPercent: 0.1)
	// We allocate 10% of the data to testing (TestPercent: 0.1)
	// We define a data processor for the four float32 data points. The name petal_sizes MUST match the name of the input defined above
	// The lineOffset is 0 because the data is the first thing in the csv row and the dataLength is 4 because there are 4 floats to train
	// The preprocessor.NewDivisor(errorHandler) will scale the floats to between 0 and 1
	// We use a preprocessor.ReadCsvFloat32s because under the hood a lineOffset: 0 dataLength: 4 will grab the first four elements of the csv row and return them as a csv string. It will convert the string to a slice of float32 values
	// We use a preprocessor.ConvertDivisorToFloat32SliceTensor to convert that slice of floats into a tensorflow Tensor. The output of this function will be passed to the model for training and evaluating
	dataset, e := data.NewSingleFileDataset(
		logger,
		errorHandler,
		"examples/iris/data/iris.data",
		cacheDir,
		4,
		0.8,
		0.1,
		0.1,
		preprocessor.NewProcessor(
			errorHandler,
			cacheDir,
			"petal_sizes",
			0,
			4,
			true,
			preprocessor.NewDivisor(errorHandler),
			nil,
			preprocessor.ReadCsvFloat32s,
			preprocessor.ConvertDivisorToFloat32SliceTensor,
		),
	)
	if e != nil {
		errorHandler.Error(e)
		return
	}

	logger.InfoF("main", "Shuffling dataset")
	// This will shuffle the data in a deterministic fashion, change 1 to time.Now().UnixNano() for a different shuffle each training session
	dataset.Shuffle(1)

	logger.InfoF("main", "Training model: %s", saveDir)

	// Train the model. "learn" and "evaluate" are important literal values and match the function names in the generated tensorflow model
	// Most of this should look familiar to anyone who has used tensorflow/keras
	// The key points are:
	//      We pass the data through 10 times (Epochs: 10)
	//      We enable validation, which will evaluate the model on the validation portion of the dataset above (Validation: true)
	//      We continuously (and concurrently) pre-fetch 10 batches to speed up training, though with 150 samples this has almost no effect
	// 		We calculate the accuracy of the model on training and validation datasets (metric.SparseCategoricalAccuracy)
	//		We log the training results to stdout (Verbose:1, callback.Logger)
	//		We save the best model based on the accuracy metric at the end of the validation stage of each epoch (callback.Checkpoint)
	m.Fit(
		"learn",
		"evaluate",
		dataset,
		model.FitConfig{
			Epochs:     10,
			Validation: true,
			BatchSize:  3,
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
					FileLogger: logger,
				},
				&callback.Checkpoint{
					OnEvent:    callback.EventEnd,
					OnMode:     callback.ModeVal,
					MetricName: "val_acc",
					Compare:    callback.CheckpointCompareMax,
					SaveDir:    saveDir,
				},
			},
		},
	)

	logger.InfoF("main", "Finished")

	/*
		Example output:
			2021-12-06 12:10:32.183 : log.go:147 : Logger initialised
			2021-12-06 12:10:35.358 : data.go:75 : Initialising single file dataset at: examples/iris/data/iris.data
			2021-12-06 12:10:35.366 : data.go:149 : Loading line offsets and stats from cache file
			2021-12-06 12:10:35.366 : data.go:155 : Found 151 rows. Got class counts: map[int]int{0:50, 1:50, 2:50}
			2021-12-06 12:10:35.369 : data.go:261 : Loaded Pre-Processor: petal_sizes
			2021-12-06 12:10:35.370 : data.go:269 : Loaded All Pre-Processors
			2021-12-06 12:10:35.371 : main.go:94 : Shuffling dataset
			2021-12-06 12:10:35.371 : main.go:97 : Training model: examples/iris/saved_models/trained_model
			2021-12-06 12:10:35.540 : logger.go:102 : End 1 5/5 (0s/0s) loss: 1.0395 acc: 0.0000 val_loss: 1.0090 val_acc: 0.0000
			2021-12-06 12:10:35.591 : logger.go:102 : End 2 5/5 (0s/0s) loss: 0.8690 acc: 0.1894 val_loss: 0.7775 val_acc: 0.4667
			2021-12-06 12:10:35.646 : logger.go:79 : Saved
			2021-12-06 12:10:35.693 : logger.go:102 : End 3 5/5 (0s/0s) loss: 0.6382 acc: 0.6212 val_loss: 0.5085 val_acc: 0.6667
			2021-12-06 12:10:35.712 : logger.go:79 : Saved
			2021-12-06 12:10:35.759 : logger.go:102 : End 4 5/5 (0s/0s) loss: 0.4717 acc: 0.7045 val_loss: 0.3765 val_acc: 0.6667
			2021-12-06 12:10:35.802 : logger.go:102 : End 5 5/5 (0s/0s) loss: 0.3755 acc: 0.8333 val_loss: 0.3027 val_acc: 0.8667
			2021-12-06 12:10:35.823 : logger.go:79 : Saved
			2021-12-06 12:10:35.869 : logger.go:102 : End 6 5/5 (0s/0s) loss: 0.3004 acc: 0.8864 val_loss: 0.2368 val_acc: 0.8667
			2021-12-06 12:10:35.913 : logger.go:102 : End 7 5/5 (0s/0s) loss: 0.2385 acc: 0.9091 val_loss: 0.1730 val_acc: 1.0000
			2021-12-06 12:10:35.935 : logger.go:79 : Saved
			2021-12-06 12:10:35.980 : logger.go:102 : End 8 5/5 (0s/0s) loss: 0.1930 acc: 0.9318 val_loss: 0.1272 val_acc: 1.0000
			2021-12-06 12:10:36.026 : logger.go:102 : End 9 5/5 (1s/1s) loss: 0.1644 acc: 0.9318 val_loss: 0.0984 val_acc: 1.0000
			2021-12-06 12:10:36.073 : logger.go:102 : End 10 5/5 (0s/0s) loss: 0.1471 acc: 0.9394 val_loss: 0.0801 val_acc: 1.0000
			2021-12-06 12:10:36.074 : main.go:131 : Finished
	*/
}

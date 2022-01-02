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
	saveDir := filepath.Join("../../logs", fmt.Sprintf("iris-%d", time.Now().Unix()))
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

	// Where the cached tokenizers and divisors will go, if you change your data you'll need to clear this
	cacheDir := "training-cache"

	// Create a dataset for training and evaluation. iris.data is in the format: float32, float32, float32, float32, className
	// This means our categoryOffset is 4. The dataset will automatically pass this value in as the label Tensor when training and evaluating
	// If the category is not an int, a tokenizer will be created to automatically convert string categories to ints in a sparse categorical format
	// We allocate 80% of the data to training (TrainPercent: 0.8)
	// We allocate 10% of the data to validation (ValPercent: 0.1)
	// We allocate 10% of the data to testing (TestPercent: 0.1)
	// We define a data processor for the four float32 data points. The name will be used for the tokenizer or divisor cache file
	// The lineOffset is 0 because the data is the first thing in the csv row and the dataLength is 4 because there are 4 floats to train
	// The preprocessor.NewDivisor(errorHandler) will scale the floats to between 0 and 1
	// We use a preprocessor.ReadCsvFloat32s because under the hood a lineOffset: 0 dataLength: 4 will grab the first four elements of the csv row and return them as a csv string. It will convert the string to a slice of float32 values
	// We use a preprocessor.ConvertDivisorToFloat32SliceTensor to convert that slice of floats into a tensorflow Tensor. The output of this function will be passed to the model for training and evaluating
	dataset, e := data.NewSingleFileDataset(
		logger,
		errorHandler,
		data.SingleFileDatasetConfig{
			FilePath:          "data/iris.data",
			CacheDir:          cacheDir,
			TrainPercent:      0.8,
			ValPercent:        0.1,
			TestPercent:       0.1,
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
				CacheDir:    cacheDir,
				LineOffset:  0,
				DataLength:  4,
				RequiresFit: true,
				Divisor:     preprocessor.NewDivisor(errorHandler),
				Reader:      preprocessor.ReadCsvFloat32s,
				Converter:   preprocessor.ConvertDivisorToFloat32SliceTensor,
			},
		),
	)
	if e != nil {
		errorHandler.Error(e)
		return
	}

	// This will save our divisor under savePath
	e = dataset.SaveProcessors(saveDir)
	if e != nil {
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
	e = m.CompileAndLoad(model.LossSparseCategoricalCrossentropy, optimizer.Adam(), saveDir)
	if e != nil {
		return
	}

	logger.InfoF("main", "Training model: %s", saveDir)

	// Train the model.
	// Most of this should look familiar to anyone who has used tensorflow/keras
	// The key points are:
	//      We pass the data through 10 times (Epochs: 10)
	//      We enable validation, which will evaluate the model on the validation portion of the dataset above (Validation: true)
	//      We continuously (and concurrently) pre-fetch 10 batches to speed up training, though with 150 samples this has almost no effect
	// 		We calculate the accuracy of the model on training and validation datasets (metric.SparseCategoricalAccuracy)
	//		We log the training results to stdout (Verbose:1, callback.Logger)
	//		We save the best model based on the accuracy metric at the end of the validation stage of each epoch (callback.Checkpoint)
	m.Fit(
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
				&callback.RecordStats{
					OnEvent:        callback.EventEnd,
					OnMode:         callback.ModeVal,
					RecordDir:      saveDir,
					RecordFileName: "train_stats.csv",
				},
				&callback.RecordStats{
					OnEvent:        callback.EventSave,
					OnMode:         callback.ModeVal,
					RecordDir:      saveDir,
					RecordFileName: "saved_stats.csv",
				},
			},
		},
	)

	m, e = model.LoadModel(errorHandler, logger, saveDir)
	if e != nil {
		return
	}

	logger.InfoF("main", "Finished training")

	// Create an inference provider, with a processor which will accept our input of [][]float32 and turn it into a tensor
	// We pass in the location of the processors we saved above in dataset.SaveProcessors
	// Note that the name of the processor must match the name used in the dataset above, as that will load the correct divisor config
	inference, e := data.NewInference(
		logger,
		errorHandler,
		saveDir,
		preprocessor.NewProcessor(
			errorHandler,
			"petal_sizes",
			preprocessor.ProcessorConfig{
				Converter: preprocessor.ConvertDivisorToFloat32SliceTensor,
			},
		),
	)
	if e != nil {
		return
	}

	// This will take our input and pass it through the processors defined above to create tensors
	// Note that we are passing in a [][]float32 as m.Predict is designed to be able to predict on multiple samples
	inputTensors, e := inference.GenerateInputs([][]float32{{6.0, 3.0, 4.8, 1.8}})
	if e != nil {
		return
	}

	// Predict the class of the input (should be Iris-virginica / 2)
	// Note that due to the automatic conversion of the labels in the dataset the classes are: Iris-setosa: 0, Iris-versicolor: 1, Iris-virginica: 2
	// These are the order of the classes in the unshuffled csv dataset
	outputTensor, e := m.Predict(inputTensors...)
	if e != nil {
		return
	}

	// Cast the tensor to [][]float32
	outputValues := outputTensor.Value().([][]float32)

	logger.InfoF(
		"main",
		"Predicted classes: %s: %f, %s: %f, %s: %f",
		"Iris-setosa",
		outputValues[0][0],
		"Iris-versicolor",
		outputValues[0][1],
		"Iris-virginica",
		outputValues[0][2],
	)

	/*
		Example output:
			2021-12-07 06:16:33.676 : log.go:147 : Logger initialised
			Initialising model
			Tracing learn
			Tracing evaluate
			Tracing predict
			Saving model
			Completed model base
			2021-12-07 06:16:37.111 : single_file_dataset.go:66 : Initialising single file dataset at: data/iris.data
			2021-12-07 06:16:37.115 : single_file_dataset.go:140 : Loading line offsets and stats from cache file
			2021-12-07 06:16:37.116 : single_file_dataset.go:146 : Found 151 rows. Got class counts: map[int]int{0:50, 1:50, 2:50}
			2021-12-07 06:16:37.117 : single_file_dataset.go:253 : Loaded Pre-Processor: petal_sizes
			2021-12-07 06:16:37.118 : single_file_dataset.go:261 : Loaded All Pre-Processors
			2021-12-07 06:16:37.118 : main.go:101 : Shuffling dataset
			2021-12-07 06:16:37.119 : main.go:105 : Training model: ../../logs/iris-
			2021-12-07 06:16:37.301 : logger.go:102 : End 1 5/5 (0s/0s) loss: 1.0304 acc: 0.0000 val_loss: 0.9951 val_acc: 0.0000
			2021-12-07 06:16:37.365 : logger.go:102 : End 2 5/5 (0s/0s) loss: 0.8511 acc: 0.2348 val_loss: 0.7440 val_acc: 0.6000
			2021-12-07 06:16:37.423 : logger.go:79 : Saved
			2021-12-07 06:16:37.470 : logger.go:102 : End 3 5/5 (0s/0s) loss: 0.6179 acc: 0.6439 val_loss: 0.4908 val_acc: 0.6667
			2021-12-07 06:16:37.490 : logger.go:79 : Saved
			2021-12-07 06:16:37.536 : logger.go:102 : End 4 5/5 (0s/0s) loss: 0.4633 acc: 0.7197 val_loss: 0.3696 val_acc: 0.6667
			2021-12-07 06:16:37.583 : logger.go:102 : End 5 5/5 (0s/0s) loss: 0.3738 acc: 0.8258 val_loss: 0.3011 val_acc: 0.8667
			2021-12-07 06:16:37.606 : logger.go:79 : Saved
			2021-12-07 06:16:37.653 : logger.go:102 : End 6 5/5 (0s/0s) loss: 0.3030 acc: 0.8864 val_loss: 0.2409 val_acc: 0.8667
			2021-12-07 06:16:37.703 : logger.go:102 : End 7 5/5 (0s/0s) loss: 0.2438 acc: 0.9015 val_loss: 0.1806 val_acc: 1.0000
			2021-12-07 06:16:37.722 : logger.go:79 : Saved
			2021-12-07 06:16:37.770 : logger.go:102 : End 8 5/5 (0s/0s) loss: 0.1987 acc: 0.9318 val_loss: 0.1348 val_acc: 1.0000
			2021-12-07 06:16:37.817 : logger.go:102 : End 9 5/5 (0s/0s) loss: 0.1689 acc: 0.9394 val_loss: 0.1048 val_acc: 1.0000
			2021-12-07 06:16:37.867 : logger.go:102 : End 10 5/5 (0s/0s) loss: 0.1500 acc: 0.9394 val_loss: 0.0851 val_acc: 1.0000
			2021-12-07 06:16:37.869 : main.go:146 : Finished training
			2021-12-07 06:16:37.895 : main.go:178 : Predicted classes: Iris-setosa: 0.000037, Iris-versicolor: 0.148679, Iris-virginica: 0.851284
	*/
}

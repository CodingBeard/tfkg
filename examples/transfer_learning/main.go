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
		layer.Dense(100).SetActivation("swish").SetName("dense_1"),
		layer.Dense(100).SetActivation("swish").SetName("dense_2"),
		layer.Dense(float64(dataset.NumCategoricalClasses())).SetActivation("softmax").SetName("dense_3"),
	)

	// This part is pretty nasty under the hood. Effectively it will generate some python code for our model and execute it to save the model in a format we can load and train
	// A python binary must be available to use for this to work
	// The batchSize used in CompileAndLoad must match the BatchSize used in Fit
	batchSize := 3
	e = m.CompileAndLoad(model.CompileConfig{
		Loss:             model.LossSparseCategoricalCrossentropy,
		Optimizer:        optimizer.Adam(),
		ModelInfoSaveDir: saveDir,
		BatchSize:        batchSize,
	})
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
			BatchSize:  batchSize,
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

	logger.InfoF("main", "Finished training")

	// Get the weights and biases from the trained model
	dense1Weights, e := m.GetLayerWeights("dense_1")
	if e != nil {
		errorHandler.Error(e)
		return
	}
	dense2Weights, e := m.GetLayerWeights("dense_2")
	if e != nil {
		errorHandler.Error(e)
		return
	}
	dense3Weights, e := m.GetLayerWeights("dense_3")
	if e != nil {
		errorHandler.Error(e)
		return
	}

	// Create a new model of the same design and pass in the weights and biases from the trained model
	transferred := model.NewSequentialModel(
		logger,
		errorHandler,
		layer.Input().SetInputShape(tf.MakeShape(-1, 4)).SetDtype(layer.Float32),
		layer.Dense(100).
			SetName("dense_1").
			SetActivation("swish").
			SetLayerWeights(dense1Weights),
		layer.Dense(100).
			SetName("dense_2").
			SetActivation("swish").
			SetLayerWeights(dense2Weights),
		layer.Dense(float64(dataset.NumCategoricalClasses())).
			SetName("dense_3").
			SetActivation("softmax").
			SetLayerWeights(dense3Weights),
	)

	// Compile the transfer model, the weights will be set during this step
	e = transferred.CompileAndLoad(model.CompileConfig{
		Loss:             model.LossSparseCategoricalCrossentropy,
		Optimizer:        optimizer.Adam(),
		ModelInfoSaveDir: transferredSaveDir,
		BatchSize:        batchSize,
	})
	if e != nil {
		return
	}

	// Evaluate the transfer model, we should see a high accuracy in the logs
	transferred.Evaluate(
		data.GeneratorModeTest,
		dataset,
		model.EvaluateConfig{
			BatchSize: batchSize,
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

	// Save the transferred model
	e = transferred.Save(transferredSaveDir)
	if e != nil {
		errorHandler.Error(e)
		return
	}
	fmt.Println()

	/*
		Example output:
			2021-12-29 17:44:56.410 : log.go:147 : Logger initialised
			2021-12-29 17:44:56.416 : single_file_dataset.go:87 : Initialising single file dataset at: data/iris.data
			2021-12-29 17:44:56.430 : single_file_dataset.go:204 : Loading line offsets and stats from cache file
			2021-12-29 17:44:56.433 : single_file_dataset.go:213 : Found 150 rows. Got class counts: map[int]int{0:50, 1:50, 2:50} Got class weights: map[int]float32{0:1, 1:1, 2:1}
			2021-12-29 17:44:56.439 : single_file_dataset.go:377 : Loaded Pre-Processor: petal_sizes
			2021-12-29 17:44:56.440 : single_file_dataset.go:385 : Loaded All Pre-Processors
			2021-12-29 17:44:56.449 : main.go:103 : Shuffling dataset
			2021-12-29 17:44:56.452 : model.go:785 : Compiling and loading model. If anything goes wrong python error messages will be printed out.
			Initialising model
			Tracing learn
			Tracing evaluate
			Tracing predict
			Tracing get_weights
			Saving model
			Completed model base
			2021-12-29 17:45:13.019 : main.go:124 : Training model: ../../logs/transfer-learning-1640799896
			2021-12-29 17:45:14.153 : logger.go:110 : End 1 5/5 (1s/1s) loss: 1.0087 acc: 0.0000 val_loss: 0.9264 val_acc: 0.0667
			2021-12-29 17:45:14.442 : logger.go:87 : Saved
			2021-12-29 17:45:14.658 : logger.go:110 : End 2 5/5 (0s/0s) loss: 0.7795 acc: 0.4667 val_loss: 0.5793 val_acc: 0.7333
			2021-12-29 17:45:14.716 : logger.go:87 : Saved
			2021-12-29 17:45:14.921 : logger.go:110 : End 3 5/5 (0s/0s) loss: 0.5271 acc: 0.6833 val_loss: 0.3516 val_acc: 0.7333
			2021-12-29 17:45:15.137 : logger.go:110 : End 4 5/5 (1s/1s) loss: 0.3928 acc: 0.7917 val_loss: 0.2500 val_acc: 0.9333
			2021-12-29 17:45:15.193 : logger.go:87 : Saved
			2021-12-29 17:45:15.407 : logger.go:110 : End 5 5/5 (0s/0s) loss: 0.3006 acc: 0.9000 val_loss: 0.1814 val_acc: 1.0000
			2021-12-29 17:45:15.464 : logger.go:87 : Saved
			2021-12-29 17:45:15.690 : logger.go:110 : End 6 5/5 (0s/0s) loss: 0.2207 acc: 0.9667 val_loss: 0.1111 val_acc: 1.0000
			2021-12-29 17:45:15.895 : logger.go:110 : End 7 5/5 (0s/0s) loss: 0.1644 acc: 0.9667 val_loss: 0.0665 val_acc: 1.0000
			2021-12-29 17:45:16.106 : logger.go:110 : End 8 5/5 (1s/1s) loss: 0.1326 acc: 0.9667 val_loss: 0.0435 val_acc: 1.0000
			2021-12-29 17:45:16.342 : logger.go:110 : End 9 5/5 (0s/0s) loss: 0.1144 acc: 0.9750 val_loss: 0.0302 val_acc: 1.0000
			2021-12-29 17:45:16.563 : logger.go:110 : End 10 5/5 (0s/0s) loss: 0.1033 acc: 0.9750 val_loss: 0.0222 val_acc: 1.0000
			2021-12-29 17:45:16.566 : main.go:177 : Finished training
			2021-12-29 17:45:16.712 : model.go:785 : Compiling and loading model. If anything goes wrong python error messages will be printed out.
			Initialising model
			Tracing learn
			Tracing evaluate
			Tracing predict
			Tracing get_weights
			Saving model
			Completed model base
			2021-12-29 17:45:32.896 : logger.go : Test 1 2/5 (0s/0s) test_loss: 0.1193 test_acc: 1.0000 | Prefetched 2
	*/
}

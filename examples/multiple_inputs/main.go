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
	saveDir := "examples/multiple_inputs/saved_models/trained_model"
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

	// Define four inputs, each going into one of four dense layers. One set for each of the data points in the iris dataset
	input1 := layer.NewInput(tf.MakeShape(-1, 1), layer.Float32)
	dense1 := layer.NewDense(
		10,
		layer.Float32,
		layer.DenseConfig{Name: "dense_1", Activation: "swish"},
	)(input1)

	input2 := layer.NewInput(tf.MakeShape(-1, 1), layer.Float32)
	dense2 := layer.NewDense(
		10,
		layer.Float32,
		layer.DenseConfig{Name: "dense_2", Activation: "swish"},
	)(input2)

	input3 := layer.NewInput(tf.MakeShape(-1, 1), layer.Float32)
	dense3 := layer.NewDense(
		10,
		layer.Float32,
		layer.DenseConfig{Name: "dense_3", Activation: "swish"},
	)(input3)

	input4 := layer.NewInput(tf.MakeShape(-1, 1), layer.Float32)
	dense4 := layer.NewDense(
		10,
		layer.Float32,
		layer.DenseConfig{Name: "dense_4", Activation: "swish"},
	)(input4)

	// Concatenate all the dense layers into a single layer
	concat := layer.NewConcatenate(-1)(dense1, dense2, dense3, dense4)

	// Pass the concatenated into a simple dense network
	denseMerged := layer.NewDense(
		100,
		layer.Float32,
		layer.DenseConfig{Name: "dense_merged", Activation: "swish"},
	)(concat)
	denseMerged2 := layer.NewDense(
		100,
		layer.Float32,
		layer.DenseConfig{Name: "dense_merged_2", Activation: "swish"},
	)(denseMerged)

	// Define the output as having three units, as there are three classes to predict
	output := layer.NewDense(
		3,
		layer.Float32,
		layer.DenseConfig{Name: "output", Activation: "softmax"},
	)(denseMerged2)

	// Define a simple keras style Functional model
	// Note that you don't need to pass in the inputs, the output variable contains all the other nodes as long as you use the same syntax of layer.New()(input)
	m := model.NewModel(
		logger,
		errorHandler,
		output,
	)

	// This part is pretty nasty under the hood. Effectively it will generate some python code for our model and execute it to save the model in a format we can load and train
	// A python binary must be available to use for this to work
	// The batchSize MUST match the batch size in the call to Fit or Evaluate
	e = m.CompileAndLoad(3)
	if e != nil {
		return
	}

	// Where the cached tokenizers and divisors will go, if you change your data you'll need to clear this
	cacheDir := "examples/multiple_inputs/training-cache"

	// Create a dataset for training and evaluation. iris.data is in the format: float32, float32, float32, float32, className
	// This means our categoryOffset is 4. The dataset will automatically pass this value in as the label Tensor when training and evaluating
	// If the category is not an int, a tokenizer will be created to automatically convert string categories to ints in a sparse categorical format
	// We allocate 80% of the data to training (TrainPercent: 0.8)
	// We allocate 10% of the data to validation (ValPercent: 0.1)
	// We allocate 10% of the data to testing (TestPercent: 0.1)
	// We define four data processors for the four float32 data points. The name will be used for the tokenizer or divisor cache file
	// The lineOffset is the offset in the data file
	// The preprocessor.NewDivisor(errorHandler) will scale the floats to between 0 and 1
	// We use a preprocessor.ReadCsvFloat32s because under the hood a lineOffset: 0 dataLength: 4 will grab the first four elements of the csv row and return them as a csv string. It will convert the string to a slice of float32 values
	// We use a preprocessor.ConvertDivisorToFloat32SliceTensor to convert that slice of floats into a tensorflow Tensor. The output of this function will be passed to the model for training and evaluating
	dataset, e := data.NewSingleFileDataset(
		logger,
		errorHandler,
		data.SingleFileDatasetConfig{
			FilePath:          "examples/iris/data/iris.data",
			CacheDir:          cacheDir,
			CategoryOffset:    4,
			TrainPercent:      0.8,
			ValPercent:        0.1,
			TestPercent:       0.1,
			IgnoreParseErrors: true,
		},
		preprocessor.NewProcessor(
			errorHandler,
			"sepal_length",
			preprocessor.ProcessorConfig{
				CacheDir:    cacheDir,
				LineOffset:  0,
				RequiresFit: true,
				Divisor:     preprocessor.NewDivisor(errorHandler),
				Reader:      preprocessor.ReadCsvFloat32s,
				Converter:   preprocessor.ConvertDivisorToFloat32SliceTensor,
			},
		),
		preprocessor.NewProcessor(
			errorHandler,
			"sepal_width",
			preprocessor.ProcessorConfig{
				CacheDir:    cacheDir,
				LineOffset:  1,
				RequiresFit: true,
				Divisor:     preprocessor.NewDivisor(errorHandler),
				Reader:      preprocessor.ReadCsvFloat32s,
				Converter:   preprocessor.ConvertDivisorToFloat32SliceTensor,
			},
		),
		preprocessor.NewProcessor(
			errorHandler,
			"petal_length",
			preprocessor.ProcessorConfig{
				CacheDir:    cacheDir,
				LineOffset:  2,
				RequiresFit: true,
				Divisor:     preprocessor.NewDivisor(errorHandler),
				Reader:      preprocessor.ReadCsvFloat32s,
				Converter:   preprocessor.ConvertDivisorToFloat32SliceTensor,
			},
		),
		preprocessor.NewProcessor(
			errorHandler,
			"petal_width",
			preprocessor.ProcessorConfig{
				CacheDir:    cacheDir,
				LineOffset:  3,
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

	logger.InfoF("main", "Training model: %s", saveDir)

	// Train the model.
	// Most of this should look familiar to anyone who has used tensorflow/keras
	// The key points are:
	//      The batchSize MUST match the batch size in the call to CompileAndLoad
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
			},
		},
	)

	logger.InfoF("main", "Finished training")

	// You do not need to load the model right after training, but this shows the weights were saved
	m, e = model.LoadModel(errorHandler, logger, saveDir)
	if e != nil {
		errorHandler.Error(e)
		return
	}

	// Create an inference provider, with four processors which will accept our inputs of [][]float32 and turn it into a tensor
	// We pass in the location of the processors we saved above in dataset.SaveProcessors
	// Note that the name of the processor must match the name used in the dataset above, as that will load the correct divisor config
	inference, e := data.NewInference(
		logger,
		errorHandler,
		saveDir,
		preprocessor.NewProcessor(
			errorHandler,
			"sepal_length",
			preprocessor.ProcessorConfig{
				Converter: preprocessor.ConvertDivisorToFloat32SliceTensor,
			},
		),
		preprocessor.NewProcessor(
			errorHandler,
			"sepal_width",
			preprocessor.ProcessorConfig{
				Converter: preprocessor.ConvertDivisorToFloat32SliceTensor,
			},
		),
		preprocessor.NewProcessor(
			errorHandler,
			"petal_length",
			preprocessor.ProcessorConfig{
				Converter: preprocessor.ConvertDivisorToFloat32SliceTensor,
			},
		),
		preprocessor.NewProcessor(
			errorHandler,
			"petal_width",
			preprocessor.ProcessorConfig{
				Converter: preprocessor.ConvertDivisorToFloat32SliceTensor,
			},
		),
	)
	if e != nil {
		return
	}

	// This will take our inputs and pass it through the processors defined above to create tensors
	// Note that we are passing in [][]float32 values as m.Predict is designed to be able to predict on multiple samples
	inputTensors, e := inference.GenerateInputs(
		[][]float32{{6.0}},
		[][]float32{{3.0}},
		[][]float32{{4.8}},
		[][]float32{{1.8}},
	)
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
			2021-12-08 18:01:29.880 : log.go:147 : Logger initialised
			2021-12-08 18:01:29.885 : model.go:715 : Compiling and loading model. If anything goes wrong python error messages will be printed out.
			Initialising model
			Tracing learn
			Tracing evaluate
			Tracing predict
			Saving model
			Completed model base
			2021-12-08 18:01:51.506 : single_file_dataset.go:66 : Initialising single file dataset at: examples/iris/data/iris.data
			2021-12-08 18:01:51.515 : single_file_dataset.go:140 : Loading line offsets and stats from cache file
			2021-12-08 18:01:51.517 : single_file_dataset.go:146 : Found 151 rows. Got class counts: map[int]int{0:50, 1:50, 2:50}
			2021-12-08 18:01:51.520 : single_file_dataset.go:253 : Loaded Pre-Processor: sepal_length
			2021-12-08 18:01:51.522 : single_file_dataset.go:253 : Loaded Pre-Processor: sepal_width
			2021-12-08 18:01:51.524 : single_file_dataset.go:253 : Loaded Pre-Processor: petal_length
			2021-12-08 18:01:51.527 : single_file_dataset.go:253 : Loaded Pre-Processor: petal_width
			2021-12-08 18:01:51.528 : single_file_dataset.go:261 : Loaded All Pre-Processors
			2021-12-08 18:01:51.530 : main.go:187 : Shuffling dataset
			2021-12-08 18:01:51.532 : main.go:191 : Training model: examples/multiple_inputs/saved_models/trained_model
			2021-12-08 18:01:53.134 : logger.go:102 : End 1 5/5 (1s/1s) loss: 1.0580 acc: 0.0000 val_loss: 1.0550 val_acc: 0.0000
			2021-12-08 18:01:53.342 : logger.go:102 : End 2 5/5 (0s/0s) loss: 0.9135 acc: 0.0682 val_loss: 0.8033 val_acc: 0.2000
			2021-12-08 18:01:53.763 : logger.go:79 : Saved
			2021-12-08 18:01:53.974 : logger.go:102 : End 3 5/5 (0s/0s) loss: 0.6254 acc: 0.5682 val_loss: 0.4964 val_acc: 0.6667
			2021-12-08 18:01:54.023 : logger.go:79 : Saved
			2021-12-08 18:01:54.237 : logger.go:102 : End 4 5/5 (0s/0s) loss: 0.4571 acc: 0.6591 val_loss: 0.3813 val_acc: 0.6667
			2021-12-08 18:01:54.447 : logger.go:102 : End 5 5/5 (0s/0s) loss: 0.3710 acc: 0.8258 val_loss: 0.2941 val_acc: 0.8667
			2021-12-08 18:01:54.499 : logger.go:79 : Saved
			2021-12-08 18:01:54.709 : logger.go:102 : End 6 5/5 (0s/0s) loss: 0.2864 acc: 0.9091 val_loss: 0.1828 val_acc: 1.0000
			2021-12-08 18:01:54.761 : logger.go:79 : Saved
			2021-12-08 18:01:54.971 : logger.go:102 : End 7 5/5 (0s/0s) loss: 0.2162 acc: 0.9470 val_loss: 0.1189 val_acc: 1.0000
			2021-12-08 18:01:55.182 : logger.go:102 : End 8 5/5 (1s/1s) loss: 0.1735 acc: 0.9545 val_loss: 0.0837 val_acc: 1.0000
			2021-12-08 18:01:55.389 : logger.go:102 : End 9 5/5 (0s/0s) loss: 0.1454 acc: 0.9621 val_loss: 0.0662 val_acc: 1.0000
			2021-12-08 18:01:55.599 : logger.go:102 : End 10 5/5 (0s/0s) loss: 0.1281 acc: 0.9621 val_loss: 0.0579 val_acc: 1.0000
			2021-12-08 18:01:55.601 : main.go:233 : Finished training
			2021-12-08 18:01:56.864 : main.go:319 : Predicted classes: Iris-setosa: 0.000334, Iris-versicolor: 0.318527, Iris-virginica: 0.681140
	*/
}

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
	saveDir := filepath.Join("../../logs", fmt.Sprintf("sign-%d", time.Now().Unix()))
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

	// Create a dataset for training and evaluation. sign.data is in the format of ./$category/IMG_*.JPG
	// We allocate 80% of the data to training (TrainPercent: 0.8)
	// We allocate 10% of the data to validation (ValPercent: 0.1)
	// We allocate 10% of the data to testing (TestPercent: 0.1)
	// We define a data processor for the image files. The name will be used for the tokenizer or divisor cache file
	// The preprocessor.NewImage will load the image, convert it to grayscale and normalize the floats to between 0 and 1
	// We use a preprocessor.ReadJpg as the images are jpegs
	// We use a preprocessor.ConvertImageToFloat32SliceTensor to convert that image into a tensorflow Tensor. The output of this function will be passed to the model for training and evaluating
	dataset, e := data.NewImgFolderDataset(
		logger,
		errorHandler,
		data.ImgFolderDatasetConfig{
			ParentDir:    "data",
			CacheDir:     cacheDir,
			TrainPercent: 0.8,
			ValPercent:   0.1,
			TestPercent:  0.1,
		},
		preprocessor.NewProcessor(
			errorHandler,
			"sign_img",
			preprocessor.ProcessorConfig{
				CacheDir: cacheDir,
				Image: preprocessor.NewImage(errorHandler, preprocessor.ImageConfig{
					ColorMode: preprocessor.ImageColorGray,
					ResizeX:   100,
					ResizeY:   100,
				}),
				Reader:    preprocessor.ReadJpg,
				Converter: preprocessor.ConvertImageToFloat32SliceTensor,
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

	// Define a VGG inspired Sequential model with two hidden Dense layers
	m := model.NewSequentialModel(
		logger,
		errorHandler,
		layer.Input().SetInputShape(tf.MakeShape(-1, 100, 100, 1)).SetDtype(layer.Float32),
		layer.Conv2D(64, 3).SetActivation("swish"),
		layer.MaxPooling2D().SetPoolSize([]interface{}{2, 2}),
		layer.Conv2D(128, 3).SetActivation("swish"),
		layer.MaxPooling2D().SetPoolSize([]interface{}{2, 2}),
		layer.Conv2D(256, 3).SetActivation("swish"),
		layer.MaxPooling2D().SetPoolSize([]interface{}{2, 2}),
		layer.Conv2D(512, 3).SetActivation("swish"),
		layer.MaxPooling2D().SetPoolSize([]interface{}{2, 2}),
		layer.Conv2D(512, 3).SetActivation("swish"),
		layer.GlobalMaxPooling2D(),
		layer.Dense(1024).SetActivation("swish"),
		layer.Dense(1024).SetActivation("swish"),
		layer.Dense(float64(dataset.NumCategoricalClasses())).SetActivation("softmax"),
	)

	// This part is pretty nasty under the hood. Effectively it will generate some python code for our model and execute it to save the model in a format we can load and train
	// A python binary must be available to use for this to work
	// The batchSize used in CompileAndLoad must match the BatchSize used in Fit
	batchSize := 10
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

	logger.InfoF("main", "Loading best model")
	m, e = model.LoadModel(errorHandler, logger, saveDir)
	if e != nil {
		return
	}

	logger.InfoF("main", "Evaluating saved model")
	m.Evaluate(
		data.GeneratorModeTest,
		dataset,
		model.EvaluateConfig{
			BatchSize: batchSize,
			PreFetch:  10,
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
				&callback.RecordStats{
					OnEvent:        callback.EventEnd,
					OnMode:         callback.ModeTest,
					RecordDir:      saveDir,
					RecordFileName: "test_stats.csv",
				},
			},
			Verbose: 1,
		},
	)

	// Create an inference provider, with a processor which will accept our input of an image filepath as []string and turn it into a tensor
	// We pass in the location of the processors we saved above in dataset.SaveProcessors
	inference, e := data.NewInference(
		logger,
		errorHandler,
		saveDir,
		preprocessor.NewProcessor(
			errorHandler,
			"sign_img",
			preprocessor.ProcessorConfig{
				CacheDir: cacheDir,
				Image: preprocessor.NewImage(errorHandler, preprocessor.ImageConfig{
					ColorMode: preprocessor.ImageColorGray,
					ResizeX:   100,
					ResizeY:   100,
				}),
				Reader:    preprocessor.ReadJpg,
				Converter: preprocessor.ConvertImageToFloat32SliceTensor,
			},
		),
	)
	if e != nil {
		return
	}

	// This will take our input and pass it through the processors defined above to create tensors
	// Note that we are passing in a []string] as m.Predict is designed to be able to predict on multiple samples
	inputTensors, e := inference.GenerateInputs([]string{"data/0/IMG_1118.JPG"})
	if e != nil {
		return
	}

	// Predict the class of the input (should be 0)
	// These are the order of the classes in the parent directory
	outputTensor, e := m.Predict(inputTensors...)
	if e != nil {
		return
	}

	// Cast the tensor to [][]float32
	outputValues := outputTensor.Value().([][]float32)

	logger.InfoF(
		"main",
		"Predicted classes: %v",
		outputValues[0],
	)

	/*
		Example output:
			2021-12-28 09:03:18.260 : log.go:147 : Logger initialised
			2021-12-28 09:03:18.260 : img_folder_dataset.go:77 : Initialising single file dataset at: data
			2021-12-28 09:03:18.260 : img_folder_dataset.go:197 : Reading image file paths and counting stats
			2021-12-28 09:03:18.263 : img_folder_dataset.go:290 : Found 2062 images. Got class counts: map[int]int{0:205, 1:206, 2:206, 3:206, 4:207, 5:207, 6:207, 7:206, 8:208, 9:204} Got class weights: map[int]float32{0:1.0146341, 1:1.0097088, 2:1.0097088, 3:1.0097088, 4:1.004831, 5:1.004831, 6:1.004831, 7:1.0097088, 8:1, 9:1.0196079}
			2021-12-28 09:03:18.263 : main.go:93 : Shuffling dataset
			2021-12-28 09:03:18.263 : model.go:726 : Compiling and loading model. If anything goes wrong python error messages will be printed out.
			Initialising model
			Tracing learn
			Tracing evaluate
			Tracing predict
			Saving model
			Completed model base
			2021-12-28 09:03:22.658 : main.go:124 : Training model: ..\..\logs\sign-1640682198
			2021-12-28 09:03:29.716 : logger.go:110 : End 1 20/20 (4s/4s) loss: 2.2263 acc: 0.0261 val_loss: 1.6926 val_acc: 0.1500
			2021-12-28 09:03:29.903 : logger.go:87 : Saved
			2021-12-28 09:03:34.310 : logger.go:110 : End 2 20/20 (5s/5s) loss: 0.8832 acc: 0.6212 val_loss: 0.6961 val_acc: 0.7350
			2021-12-28 09:03:34.451 : logger.go:87 : Saved
			2021-12-28 09:03:38.839 : logger.go:110 : End 3 20/20 (4s/4s) loss: 0.3613 acc: 0.8594 val_loss: 0.3200 val_acc: 0.9250
			2021-12-28 09:03:38.974 : logger.go:87 : Saved
			2021-12-28 09:03:43.365 : logger.go:110 : End 4 20/20 (5s/5s) loss: 0.2760 acc: 0.8964 val_loss: 0.5694 val_acc: 0.8350
			2021-12-28 09:03:47.762 : logger.go:110 : End 5 20/20 (4s/4s) loss: 0.2732 acc: 0.9139 val_loss: 0.8170 val_acc: 0.8650
			2021-12-28 09:03:52.161 : logger.go:110 : End 6 20/20 (5s/5s) loss: 0.2662 acc: 0.9297 val_loss: 0.7916 val_acc: 0.8500
			2021-12-28 09:03:56.562 : logger.go:110 : End 7 20/20 (4s/4s) loss: 0.1304 acc: 0.9588 val_loss: 0.3928 val_acc: 0.9300
			2021-12-28 09:03:56.696 : logger.go:87 : Saved
			2021-12-28 09:04:01.102 : logger.go:110 : End 8 20/20 (5s/5s) loss: 0.0593 acc: 0.9752 val_loss: 0.5345 val_acc: 0.9300
			2021-12-28 09:04:05.502 : logger.go:110 : End 9 20/20 (4s/4s) loss: 0.1872 acc: 0.9527 val_loss: 0.8384 val_acc: 0.9000
			2021-12-28 09:04:09.912 : logger.go:110 : End 10 20/20 (4s/4s) loss: 0.4604 acc: 0.9297 val_loss: 51.5520 val_acc: 0.2250
			2021-12-28 09:04:09.912 : main.go:177 : Finished training
			2021-12-28 09:04:09.912 : main.go:179 : Loading best model
			2021-12-28 09:04:10.104 : main.go:185 : Evaluating saved model
			2021-12-28 09:04:10.448 : logger.go:110 : End 1 20/20 (0s/0s) test_loss: 0.1811 test_acc: 0.9400
			2021-12-28 09:04:10.448 : inference.go:26 : Initialising inference provider with processors loaded from: ..\..\logs\sign-1640682198
			2021-12-28 09:04:10.593 : main.go:256 : Predicted classes: [0.98045534 0.0024687306 5.1811203e-06 2.2673211e-07 0.0011776222 2.1564902e-06 0.00015993018 0.0026970028 0.013013764 2.0157724e-05]
	*/
}

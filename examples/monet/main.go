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
	"image"
	"image/color"
	"image/jpeg"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func main() {
	logsDir := filepath.Join("../../logs", fmt.Sprintf("monet-%d", time.Now().Unix()))
	e := os.MkdirAll(logsDir, os.ModePerm)
	if e != nil {
		panic(e)
	}
	generatorDir := filepath.Join("../../logs", fmt.Sprintf("monet-%d", time.Now().Unix()), "generator")
	e = os.MkdirAll(generatorDir, os.ModePerm)
	if e != nil {
		panic(e)
	}
	discriminatorDir := filepath.Join("../../logs", fmt.Sprintf("monet-%d", time.Now().Unix()), "discriminator")
	e = os.MkdirAll(discriminatorDir, os.ModePerm)
	if e != nil {
		panic(e)
	}
	discriminatorCacheDir := filepath.Join(discriminatorDir, "training-cache")
	outputDir := filepath.Join("data", "training", "1")
	e = os.MkdirAll(outputDir, os.ModePerm)
	if e != nil {
		panic(e)
	}

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

	discriminatorModel := model.NewSequentialModel(
		logger,
		errorHandler,
		layer.Input().SetInputShape(tf.MakeShape(-1, 256, 256, 3)).SetDtype(layer.Float32),
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
		layer.Dense(1).SetActivation("softmax"),
	)

	e = discriminatorModel.CompileAndLoad(model.LossBinaryCrossentropy, optimizer.Adam(), discriminatorDir)
	if e != nil {
		return
	}

	// Define a simple keras style Sequential model with two hidden Dense layers
	generatorModel := model.NewSequentialModel(
		logger,
		errorHandler,
		layer.Input().SetInputShape(tf.MakeShape(-1, 256, 256, 3)).SetDtype(layer.Float32),
		layer.Conv2D(256, 3).SetPadding("same").SetActivation("swish"),
		layer.BatchNormalization(),
		layer.Rescaling(255),
	)

	e = generatorModel.CompileAndLoad(model.LossBinaryCrossentropy, optimizer.Adam(), generatorDir)
	if e != nil {
		errorHandler.Error(e)
		return
	}

	originalImagePaths, e := filepath.Glob("data/photo_jpg/*")
	if e != nil {
		panic(e)
	}
	originalImagePaths = originalImagePaths[:300]

	monetImagePaths, e := filepath.Glob("data/training/0/*")
	if e != nil {
		panic(e)
	}

	//
	generatorDataset, e := data.NewInference(
		logger,
		errorHandler,
		generatorDir,
		preprocessor.NewProcessor(
			errorHandler,
			"original_img",
			preprocessor.ProcessorConfig{
				Image: preprocessor.NewImage(
					errorHandler,
					preprocessor.ImageConfig{
						ColorMode: preprocessor.ImageColorRGB,
					},
				),
				Reader:    preprocessor.ReadJpg,
				Converter: preprocessor.ConvertImageToFloat32SliceTensor,
			},
		),
	)
	if e != nil {
		errorHandler.Error(e)
		return
	}
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(originalImagePaths), func(i, j int) {
		originalImagePaths[i], originalImagePaths[j] = originalImagePaths[j], originalImagePaths[i]
	})
	fmt.Println()
	var originalImageBatches []*tf.Tensor
	for i := 0; i < len(originalImagePaths); i++ {
		fmt.Print(fmt.Sprintf("\rLoading image batch %d", i))
		x, e := generatorDataset.GenerateInputs([]string{originalImagePaths[i]})
		if e != nil {
			errorHandler.Error(e)
			continue
		}
		originalImageBatches = append(originalImageBatches, x[0])
	}
	fmt.Println()

	for i := 0; i < 1; i++ {
		for batchNum, batch := range originalImageBatches {
			fmt.Print(fmt.Sprintf("\rgenerating fake images (%d/%d)", batchNum, len(originalImageBatches)))
			generatedImages, e := generatorModel.Predict(batch)
			if e != nil {
				errorHandler.Error(e)
				continue
			}
			fmt.Print(fmt.Sprintf("\rsaving fake images     (%d/%d)", batchNum, len(originalImageBatches)))
			for imgNum, imgBytes := range generatedImages.Value().([][][][]float32) {
				img := image.NewRGBA(image.Rect(0, 0, 256, 256))
				for y, xRow := range imgBytes {
					for x, channels := range xRow {
						img.Set(x, y, color.RGBA{
							R: uint8(channels[0]),
							G: uint8(channels[1]),
							B: uint8(channels[2]),
						})
					}
				}
				f, e := os.Create(filepath.Join(outputDir, fmt.Sprintf("%d.jpg", batchNum*10+imgNum)))
				if e != nil {
					errorHandler.Error(e)
					continue
				}
				e = jpeg.Encode(f, img, &jpeg.Options{Quality: 100})
			}
		}
		fmt.Println()

		fakeImagePaths, e := filepath.Glob("data/training/1/*")
		if e != nil {
			panic(e)
		}

		discriminatorDataset, e := data.NewValuesDataset(
			logger,
			errorHandler,
			data.ValuesDatasetConfig{
				CacheDir:     discriminatorCacheDir,
				TrainPercent: 1,
				ValPercent:   0.1,
			},
			preprocessor.NewProcessor(
				errorHandler,
				"discriminate_img",
				preprocessor.ProcessorConfig{
					CacheDir: "",
					Image: preprocessor.NewImage(errorHandler, preprocessor.ImageConfig{
						ColorMode: preprocessor.ImageColorRGB,
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

		var mixedImagePaths []string
		{
		}

		for _, path := range monetImagePaths {
			mixedImagePaths = append(mixedImagePaths, path)
		}
		for _, path := range fakeImagePaths {
			mixedImagePaths = append(mixedImagePaths, path)
		}

		rand.Seed(time.Now().UnixNano())
		rand.Shuffle(len(mixedImagePaths), func(i, j int) {
			mixedImagePaths[i], mixedImagePaths[j] = mixedImagePaths[j], mixedImagePaths[i]
		})

		var discriminatorYValues []interface{}

		for _, path := range mixedImagePaths {
			if strings.HasPrefix(path, "data/training/0") {
				discriminatorYValues = append(discriminatorYValues, 0)
			} else {
				discriminatorYValues = append(discriminatorYValues, 1)
			}
		}

		var mixedImagePathsInterface []interface{}
		for _, path := range mixedImagePaths {
			mixedImagePathsInterface = append(mixedImagePathsInterface, path)
		}

		e = discriminatorDataset.SetValues(discriminatorYValues, mixedImagePathsInterface)
		if e != nil {
			errorHandler.Error(e)
			continue
		}

		logger.InfoF("main", "Training discriminator model")
		discriminatorModel.Fit(
			discriminatorDataset,
			model.FitConfig{
				Epochs:     1,
				Validation: true,
				BatchSize:  10,
				PreFetch:   10,
				Verbose:    1,
				Metrics: []metric.Metric{
					&metric.BinaryAccuracy{
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

		discriminatorInferenceDataset, e := data.NewInference(
			logger,
			errorHandler,
			discriminatorCacheDir,
			preprocessor.NewProcessor(
				errorHandler,
				"discriminate_img",
				preprocessor.ProcessorConfig{
					CacheDir: "",
					Image: preprocessor.NewImage(errorHandler, preprocessor.ImageConfig{
						ColorMode: preprocessor.ImageColorRGB,
					}),
					Reader:    preprocessor.ReadJpg,
					Converter: preprocessor.ConvertImageToFloat32SliceTensor,
				},
			),
		)

		var generatorTrainXValues []interface{}
		var generatorTrainYValues []interface{}

		logger.InfoF("main", "Discriminating fake images")
		for _, path := range mixedImagePaths {
			if strings.HasSuffix(path, "data/training/0") {
				continue
			}
			generatorTrainXValues = append(generatorTrainXValues, path)

			x, e := discriminatorInferenceDataset.GenerateInputs([]string{path})
			if e != nil {
				errorHandler.Error(e)
				continue
			}
			prediction, e := discriminatorModel.Predict(x[0])
			if prediction.Value().([][]float32)[0][0] > 0.5 {
				generatorTrainYValues = append(generatorTrainYValues, 1)
			} else {
				generatorTrainYValues = append(generatorTrainYValues, 0)
			}
		}

		generatorTrainDataset, e := data.NewValuesDataset(
			logger,
			errorHandler,
			data.ValuesDatasetConfig{
				CacheDir:     discriminatorCacheDir,
				TrainPercent: 1,
				ValPercent:   0.1,
			},
			preprocessor.NewProcessor(
				errorHandler,
				"generate_img",
				preprocessor.ProcessorConfig{
					CacheDir: "",
					Image: preprocessor.NewImage(errorHandler, preprocessor.ImageConfig{
						ColorMode: preprocessor.ImageColorRGB,
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

		e = generatorTrainDataset.SetValues(generatorTrainYValues, generatorTrainXValues)
		if e != nil {
			errorHandler.Error(e)
			continue
		}

		logger.InfoF("main", "Training generator model")
		generatorModel.Fit(
			generatorTrainDataset,
			model.FitConfig{
				Epochs:     1,
				Validation: true,
				BatchSize:  10,
				PreFetch:   10,
				Verbose:    1,
				Metrics: []metric.Metric{
					&metric.BinaryAccuracy{
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
	}
	fmt.Println()

	//// Define a simple keras style Sequential model with two hidden Dense layers
	//m := model.NewSequentialModel(
	//	logger,
	//	errorHandler,
	//	layer.Input().SetInputShape(tf.MakeShape(-1, 4)).SetDtype(layer.Float32),
	//	layer.Dense(100).SetActivation("swish"),
	//	layer.Dense(100).SetActivation("swish"),
	//	layer.Dense(1).SetActivation("softmax"),
	//)
	//
	//// This part is pretty nasty under the hood. Effectively it will generate some python code for our model and execute it to save the model in a format we can load and train
	//// A python binary must be available to use for this to work
	//e = m.CompileAndLoad(model.LossSparseCategoricalCrossentropy, optimizer.Adam(), logsDir)
	//if e != nil {
	//	return
	//}
	//
	//logger.InfoF("main", "Training model")
	//
	//// Train the model.
	//// Most of this should look familiar to anyone who has used tensorflow/keras
	//// The key points are:
	////      We pass the data through 10 times (Epochs: 10)
	////      We enable validation, which will evaluate the model on the validation portion of the dataset above (Validation: true)
	////      We continuously (and concurrently) pre-fetch 10 batches to speed up training, though with 150 samples this has almost no effect
	//// 		We calculate the accuracy of the model on training and validation datasets (metric.SparseCategoricalAccuracy)
	////		We log the training results to stdout (Verbose:1, callback.Logger)
	//m.Fit(
	//	dataset,
	//	model.FitConfig{
	//		Epochs:     10,
	//		Validation: true,
	//		BatchSize:  1000,
	//		PreFetch:   10,
	//		Verbose:    1,
	//		Metrics: []metric.Metric{
	//			&metric.SparseCategoricalAccuracy{
	//				Name:       "acc",
	//				Confidence: 0.5,
	//				Average:    true,
	//			},
	//		},
	//		Callbacks: []callback.Callback{
	//			&callback.Logger{
	//				FileLogger:     logger,
	//				Progress:       true,
	//				ProgressLogDir: logsDir,
	//			},
	//		},
	//	},
	//)

	logger.InfoF("main", "Finished training")

}

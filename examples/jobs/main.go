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
	saveDir := filepath.Join("../../logs", fmt.Sprintf("jobs-%d", time.Now().Unix()))
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

	// We define data processors for the title, location, department, company_profile, description, and requirements. These names will be used for the tokenizer or divisor cache file
	// The lineOffset is the offset in the data file
	// The preprocessor.NewTokenizer will tokenize the strings into ints
	// We use a preprocessor.ReadStringNop because the inputs are already in the format tokenizers accept, a string
	// We use a preprocessor.ConvertTokenizerToFloat32SliceTensor to convert the output of the tokenizer to a slice of floats into a tensorflow Tensor. The output of this function will be passed to the model for training and evaluating
	titleProcessor := preprocessor.NewProcessor(
		errorHandler,
		"title",
		preprocessor.ProcessorConfig{
			CacheDir:    cacheDir,
			LineOffset:  1,
			RequiresFit: true,
			Tokenizer:   preprocessor.NewTokenizer(errorHandler, 10, 1000),
			Reader:      preprocessor.ReadStringNop,
			Converter:   preprocessor.ConvertTokenizerToFloat32SliceTensor,
		},
	)

	locationProcessor := preprocessor.NewProcessor(
		errorHandler,
		"location",
		preprocessor.ProcessorConfig{
			CacheDir:    cacheDir,
			LineOffset:  2,
			RequiresFit: true,
			Tokenizer:   preprocessor.NewTokenizer(errorHandler, 10, 1000),
			Reader:      preprocessor.ReadStringNop,
			Converter:   preprocessor.ConvertTokenizerToFloat32SliceTensor,
		},
	)

	departmentProcessor := preprocessor.NewProcessor(
		errorHandler,
		"department",
		preprocessor.ProcessorConfig{
			CacheDir:    cacheDir,
			LineOffset:  3,
			RequiresFit: true,
			Tokenizer:   preprocessor.NewTokenizer(errorHandler, 10, 1000),
			Reader:      preprocessor.ReadStringNop,
			Converter:   preprocessor.ConvertTokenizerToFloat32SliceTensor,
		},
	)

	companyProfileProcessor := preprocessor.NewProcessor(
		errorHandler,
		"company_profile",
		preprocessor.ProcessorConfig{
			CacheDir:    cacheDir,
			LineOffset:  5,
			RequiresFit: true,
			Tokenizer:   preprocessor.NewTokenizer(errorHandler, 100, 1000),
			Reader:      preprocessor.ReadStringNop,
			Converter:   preprocessor.ConvertTokenizerToFloat32SliceTensor,
		},
	)

	descriptionProcessor := preprocessor.NewProcessor(
		errorHandler,
		"description",
		preprocessor.ProcessorConfig{
			CacheDir:    cacheDir,
			LineOffset:  6,
			RequiresFit: true,
			Tokenizer:   preprocessor.NewTokenizer(errorHandler, 100, 1000),
			Reader:      preprocessor.ReadStringNop,
			Converter:   preprocessor.ConvertTokenizerToFloat32SliceTensor,
		},
	)

	requirementsProcessor := preprocessor.NewProcessor(
		errorHandler,
		"requirements",
		preprocessor.ProcessorConfig{
			CacheDir:    cacheDir,
			LineOffset:  7,
			RequiresFit: true,
			Tokenizer:   preprocessor.NewTokenizer(errorHandler, 100, 1000),
			Reader:      preprocessor.ReadStringNop,
			Converter:   preprocessor.ConvertTokenizerToFloat32SliceTensor,
		},
	)

	// Create a dataset for training and evaluation. The dataset is in the format: job_id,title,location,department,salary_range,company_profile,description,requirements,benefits,telecommuting,has_company_logo,has_questions,employment_type,required_experience,required_education,industry,function,fraudulent
	// Our categoryOffset is 17 as we are predicting whether the posting is fraudulent. The dataset will automatically pass this value in as the label Tensor when training and evaluating
	// We allocate 80% of the data to training (TrainPercent: 0.8)
	// We allocate 10% of the data to validation (ValPercent: 0.1)
	// We allocate 10% of the data to testing (TestPercent: 0.1)
	// We pass in the data processors we defined above
	dataset, e := data.NewSingleFileDataset(
		logger,
		errorHandler,
		data.SingleFileDatasetConfig{
			FilePath:          "data/fake_job_postings.csv",
			CacheDir:          cacheDir,
			TrainPercent:      0.8,
			ValPercent:        0.1,
			TestPercent:       0.1,
			IgnoreParseErrors: false,
			SkipHeaders:       true,
		},
		preprocessor.NewBinaryYProcessor(
			errorHandler,
			cacheDir,
			17,
		),
		titleProcessor,
		locationProcessor,
		departmentProcessor,
		companyProfileProcessor,
		descriptionProcessor,
		requirementsProcessor,
	)
	if e != nil {
		errorHandler.Error(e)
		return
	}

	// This will save our tokenizers under savePath
	e = dataset.SaveProcessors(saveDir)
	if e != nil {
		return
	}

	logger.InfoF("main", "Shuffling dataset")
	// This will shuffle the data in a deterministic fashion, change 1 to time.Now().UnixNano() for a different shuffle each training session
	dataset.Shuffle(1)

	// Define 6 input paths, one for each of our inputs. They will be passed into an embedding layer and then a LSTM layer
	titleInput := layer.Input().
		SetInputShape(tf.MakeShape(-1, int64(titleProcessor.Tokenizer().MaxLen()))).
		SetDtype(layer.Float32).
		SetName("title_input")

	titleEmbedding := layer.Embedding(
		float64(titleProcessor.Tokenizer().NumWords()+1),
		32,
	).
		SetName("title_embedding").
		SetInputs(titleInput)

	titleLSTM := layer.LSTM(32).
		SetName("title_lstm").
		SetInputs(titleEmbedding)

	locationInput := layer.Input().
		SetInputShape(tf.MakeShape(-1, int64(locationProcessor.Tokenizer().MaxLen()))).
		SetDtype(layer.Float32).
		SetName("location_input")

	locationEmbedding := layer.Embedding(
		float64(locationProcessor.Tokenizer().NumWords()+1),
		32,
	).
		SetName("location_embedding").
		SetInputs(locationInput)

	locationLSTM := layer.LSTM(32).
		SetName("location_lstm").
		SetInputs(locationEmbedding)

	departmentInput := layer.Input().
		SetInputShape(tf.MakeShape(-1, int64(departmentProcessor.Tokenizer().MaxLen()))).
		SetDtype(layer.Float32).
		SetName("department_input")

	departmentEmbedding := layer.Embedding(
		float64(departmentProcessor.Tokenizer().NumWords()+1),
		32,
	).
		SetName("department_embedding").
		SetInputs(departmentInput)

	departmentLSTM := layer.LSTM(32).
		SetName("department_lstm").
		SetInputs(departmentEmbedding)

	companyProfileInput := layer.Input().
		SetInputShape(tf.MakeShape(-1, int64(companyProfileProcessor.Tokenizer().MaxLen()))).
		SetDtype(layer.Float32).
		SetName("companyProfile_input")

	companyProfileEmbedding := layer.Embedding(
		float64(companyProfileProcessor.Tokenizer().NumWords()+1),
		32,
	).
		SetName("company_profile_embedding").
		SetInputs(companyProfileInput)

	companyProfileLSTM := layer.LSTM(32).
		SetName("company_profile_lstm").
		SetInputs(companyProfileEmbedding)

	descriptionInput := layer.Input().
		SetInputShape(tf.MakeShape(-1, int64(descriptionProcessor.Tokenizer().MaxLen()))).
		SetDtype(layer.Float32).
		SetName("description_input")

	descriptionEmbedding := layer.Embedding(
		float64(descriptionProcessor.Tokenizer().NumWords()+1),
		32,
	).
		SetName("description_embedding").
		SetInputs(descriptionInput)

	descriptionLSTM := layer.LSTM(32).
		SetName("description_lstm").
		SetInputs(descriptionEmbedding)

	requirementsInput := layer.Input().
		SetInputShape(tf.MakeShape(-1, int64(requirementsProcessor.Tokenizer().MaxLen()))).
		SetDtype(layer.Float32).
		SetName("requirements_input")

	requirementsEmbedding := layer.Embedding(
		float64(requirementsProcessor.Tokenizer().NumWords()+1),
		32,
	).
		SetName("requirements_embedding").
		SetInputs(requirementsInput)

	requirementsLSTM := layer.LSTM(32).
		SetName("requirements_lstm").
		SetInputs(requirementsEmbedding)

	// Merge our LSTM layers into a single tensor
	concatenate := layer.Concatenate().
		SetInputs(titleLSTM, locationLSTM, departmentLSTM, companyProfileLSTM, descriptionLSTM, requirementsLSTM)

	// Feed the merged input into a dense network
	mergedDense1 := layer.Dense(100).
		SetName("merged_dense_1").
		SetActivation("swish").
		SetInputs(concatenate)
	mergedDense2 := layer.Dense(100).
		SetName("merged_dense_2").
		SetActivation("swish").
		SetInputs(mergedDense1)

	output := layer.Dense(1).
		SetName("output").
		SetActivation("sigmoid").
		SetInputs(mergedDense2)

	// Define a keras style Functional model
	// Note that you don't need to pass in the inputs, the output variable contains all the other nodes as long as you use the same syntax of layer.New()(input)
	m := model.NewModel(
		logger,
		errorHandler,
		output,
	)

	// This part is pretty nasty under the hood. Effectively it will generate some python code for our model and execute it to save the model in a format we can load and train
	// A python binary must be available to use for this to work
	// The batchSize used in CompileAndLoad must match the BatchSize used in Fit
	batchSize := 200
	e = m.CompileAndLoad(model.CompileConfig{
		Loss:             model.LossBinaryCrossentropy,
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
	//      The batchSize MUST match the batch size in the call to CompileAndLoad
	//      We pass the data through 10 times (Epochs: 10)
	//      We enable validation, which will evaluate the model on the validation portion of the dataset above (Validation: true)
	//      We continuously (and concurrently) pre-fetch 10 batches to speed up training, though with 150 samples this has almost no effect
	// 		We calculate the accuracy of the model on training and validation datasets (metric.BinaryAccuracy)
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

	// You do not need to load the model right after training, but this shows the weights were saved
	m, e = model.LoadModel(errorHandler, logger, saveDir)
	if e != nil {
		errorHandler.Error(e)
		return
	}

	// Create an inference provider, with six processors which will accept our inputs of []string and turn them into tensors
	// We pass in the names of the processors we saved above in dataset.SaveProcessors
	// Note that the name of the processor must match the name used in the dataset above, as that will load the correct config
	inference, e := data.NewInference(
		logger,
		errorHandler,
		saveDir,
		preprocessor.NewProcessor(
			errorHandler,
			"title",
			preprocessor.ProcessorConfig{
				Converter: preprocessor.ConvertTokenizerToFloat32SliceTensor,
			},
		),
		preprocessor.NewProcessor(
			errorHandler,
			"location",
			preprocessor.ProcessorConfig{
				Converter: preprocessor.ConvertTokenizerToFloat32SliceTensor,
			},
		),
		preprocessor.NewProcessor(
			errorHandler,
			"department",
			preprocessor.ProcessorConfig{
				Converter: preprocessor.ConvertTokenizerToFloat32SliceTensor,
			},
		),
		preprocessor.NewProcessor(
			errorHandler,
			"company_profile",
			preprocessor.ProcessorConfig{
				Converter: preprocessor.ConvertTokenizerToFloat32SliceTensor,
			},
		),
		preprocessor.NewProcessor(
			errorHandler,
			"description",
			preprocessor.ProcessorConfig{
				Converter: preprocessor.ConvertTokenizerToFloat32SliceTensor,
			},
		),
		preprocessor.NewProcessor(
			errorHandler,
			"requirements",
			preprocessor.ProcessorConfig{
				Converter: preprocessor.ConvertTokenizerToFloat32SliceTensor,
			},
		),
	)
	if e != nil {
		return
	}

	// This will take our input and pass it through the processors defined above to create tensors
	// Note that we are passing in []string values as m.Predict is designed to be able to predict on multiple samples
	inputTensors, e := inference.GenerateInputs(
		[]string{
			"Marketing Intern",
		},
		[]string{
			"US, NY, New York",
		},
		[]string{
			"Marketing",
		},
		[]string{
			"We're Food52, and we've created a groundbreaking and award-winning cooking site. We support, connect, and celebrate home cooks, and give them everything they need in one place.We have a top editorial, business, and engineering team. We're focused on using technology to find new and better ways to connect people around their specific food interests, and to offer them superb, highly curated information about food and cooking. We attract the most talented home cooks and contributors in the country; we also publish well-known professionals like Mario Batali, Gwyneth Paltrow, and Danny Meyer. And we have partnerships with Whole Foods Market and Random House.Food52 has been named the best food website by the James Beard Foundation and IACP, and has been featured in the New York Times, NPR, Pando Daily, TechCrunch, and on the Today Show.We're located in Chelsea, in New York City.",
		},
		[]string{
			"Food52, a fast-growing, James Beard Award-winning online food community and crowd-sourced and curated recipe hub, is currently interviewing full- and part-time unpaid interns to work in a small team of editors, executives, and developers in its New York City headquarters.Reproducing and/or repackaging existing Food52 content for a number of partner sites, such as Huffington Post, Yahoo, Buzzfeed, and more in their various content management systemsResearching blogs and websites for the Provisions by Food52 Affiliate ProgramAssisting in day-to-day affiliate program support, such as screening affiliates and assisting in any affiliate inquiriesSupporting with PR &amp; Events when neededHelping with office administrative work, such as filing, mailing, and preparing for meetingsWorking with developers to document bugs and suggest improvements to the siteSupporting the marketing and executive staff",
		},
		[]string{
			"Experience with content management systems a major plus (any blogging counts!)Familiar with the Food52 editorial voice and aestheticLoves food, appreciates the importance of home cooking and cooking with the seasonsMeticulous editor, perfectionist, obsessive attention to detail, maddened by typos and broken links, delighted by finding and fixing themCheerful under pressureExcellent communication skillsA+ multi-tasker and juggler of responsibilities big and smallInterested in and engaged with social media like Twitter, Facebook, and PinterestLoves problem-solving and collaborating to drive Food52 forwardThinks big picture but pitches in on the nitty gritty of running a small company (dishes, shopping, administrative support)Comfortable with the realities of working for a startup: being on call on evenings and weekends, and working long hours",
		},
	)
	if e != nil {
		return
	}

	// Predict the class of the input (should be 0/non-fraudulent)
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
			2022-01-02 19:53:58.245 : log.go:147 : Logger initialised
			2022-01-02 19:53:58.246 : single_file_dataset.go:86 : Initialising single file dataset at: data/fake_job_postings.csv
			2022-01-02 19:53:58.246 : single_file_dataset.go:205 : Reading line offsets and counting stats
			2022-01-02 19:53:58.544 : single_file_dataset.go:353 : Found 17570 rows. Got class counts: map[int]int{0:17013, 1:866} Got class weights: map[int]float32{0:1, 1:19.645496}
			2022-01-02 19:53:58.544 : single_file_dataset.go:384 : Fitting Pre-Processors
			Fitting preprocessors: 6893 6893/s2022-01-02 19:53:59.679 : single_file_dataset.go:191 : Loading line offsets and stats from cache file
			2022-01-02 19:53:59.679 : single_file_dataset.go:200 : Found 17570 rows. Got class counts: map[int]int{0:17013, 1:866} Got class weights: map[int]float32{0:1, 1:19.645496}
			2022-01-02 19:53:59.683 : single_file_dataset.go:431 : Fit tokenizers
			2022-01-02 19:53:59.687 : main.go:173 : Shuffling dataset
			2022-01-02 19:53:59.688 : model.go:938 : Compiling and loading model. If anything goes wrong python error messages will be printed out.
			Initialising model
			Tracing learn
			Tracing evaluate
			Tracing predict
			Tracing get_weights
			Tracing set_weights
			Saving model
			Completed model base
			2022-01-02 19:54:28.534 : main.go:308 : Training model: ..\..\logs\jobs-1641153238
			2022-01-02 19:55:29.235 : logger.go:110 : End 1 8/8 (56s/56s) loss: 0.5936 acc: 0.7513 val_loss: 0.4469 val_acc: 0.7775
			2022-01-02 19:55:29.781 : logger.go:87 : Saved
			2022-01-02 19:56:25.185 : logger.go:110 : End 2 8/8 (55s/55s) loss: 0.2576 acc: 0.8818 val_loss: 0.1751 val_acc: 0.9238
			2022-01-02 19:56:25.208 : logger.go:87 : Saved
			2022-01-02 19:57:20.833 : logger.go:110 : End 3 8/8 (54s/54s) loss: 0.1384 acc: 0.9451 val_loss: 0.0916 val_acc: 0.9600
			2022-01-02 19:57:20.863 : logger.go:87 : Saved
			2022-01-02 19:58:16.136 : logger.go:110 : End 4 8/8 (55s/55s) loss: 0.0885 acc: 0.9664 val_loss: 0.0831 val_acc: 0.9744
			2022-01-02 19:58:16.163 : logger.go:87 : Saved
			2022-01-02 19:59:10.249 : logger.go:110 : End 5 8/8 (54s/54s) loss: 0.0760 acc: 0.9747 val_loss: 0.1354 val_acc: 0.9556
			2022-01-02 20:00:04.542 : logger.go:110 : End 6 8/8 (53s/53s) loss: 0.0964 acc: 0.9646 val_loss: 0.1370 val_acc: 0.9575
			2022-01-02 20:00:58.861 : logger.go:110 : End 7 8/8 (53s/53s) loss: 0.0810 acc: 0.9711 val_loss: 0.1074 val_acc: 0.9712
			2022-01-02 20:01:53.341 : logger.go:110 : End 8 8/8 (54s/54s) loss: 0.0375 acc: 0.9875 val_loss: 0.1201 val_acc: 0.9725
			2022-01-02 20:02:47.845 : logger.go:110 : End 9 8/8 (53s/53s) loss: 0.0269 acc: 0.9911 val_loss: 0.1365 val_acc: 0.9763
			2022-01-02 20:02:47.870 : logger.go:87 : Saved
			2022-01-02 20:03:43.072 : logger.go:110 : End 10 8/8 (55s/55s) loss: 0.0192 acc: 0.9948 val_loss: 0.1616 val_acc: 0.9769
			2022-01-02 20:03:43.098 : logger.go:87 : Saved
			2022-01-02 20:03:43.099 : main.go:362 : Finished training
			2022-01-02 20:03:45.005 : inference.go:26 : Initialising inference provider with processors loaded from: ..\..\logs\jobs-1641153238
			2022-01-02 20:03:46.422 : main.go:460 : Predicted classes: [7.0848978e-12]
	*/
}

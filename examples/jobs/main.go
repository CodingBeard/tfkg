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
	saveDir := "examples/jobs/saved_models/trained_model"
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
	cacheDir := "examples/jobs/training-cache"

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
			FilePath:          "examples/jobs/data/fake_job_postings.csv",
			CacheDir:          cacheDir,
			CategoryOffset:    17,
			TrainPercent:      0.8,
			ValPercent:        0.1,
			TestPercent:       0.1,
			IgnoreParseErrors: false,
			SkipHeaders:       true,
		},
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
	titleInput := layer.NewInput(
		tf.MakeShape(-1, int64(titleProcessor.Tokenizer().MaxLen())),
		layer.Float32,
		layer.InputConfig{Name: "title_input"},
	)

	titleEmbedding := layer.NewEmbedding(
		titleProcessor.Tokenizer().NumWords()+1,
		32,
		layer.EmbeddingConfig{Name: "title_embedding"},
	)(titleInput)

	titleLSTM := layer.NewLSTM(32, layer.LSTMConfig{Name: "title_lstm"})(titleEmbedding)

	locationInput := layer.NewInput(
		tf.MakeShape(-1, int64(locationProcessor.Tokenizer().MaxLen())),
		layer.Float32,
		layer.InputConfig{Name: "location_input"},
	)

	locationEmbedding := layer.NewEmbedding(
		locationProcessor.Tokenizer().NumWords()+1,
		32,
		layer.EmbeddingConfig{Name: "location_embedding"},
	)(locationInput)

	locationLSTM := layer.NewLSTM(32, layer.LSTMConfig{Name: "location_lstm"})(locationEmbedding)

	departmentInput := layer.NewInput(
		tf.MakeShape(-1, int64(departmentProcessor.Tokenizer().MaxLen())),
		layer.Float32,
		layer.InputConfig{Name: "department_input"},
	)

	departmentEmbedding := layer.NewEmbedding(
		departmentProcessor.Tokenizer().NumWords()+1,
		32,
		layer.EmbeddingConfig{Name: "department_embedding"},
	)(departmentInput)

	departmentLSTM := layer.NewLSTM(32, layer.LSTMConfig{Name: "department_lstm"})(departmentEmbedding)

	companyProfileInput := layer.NewInput(
		tf.MakeShape(-1, int64(companyProfileProcessor.Tokenizer().MaxLen())),
		layer.Float32,
		layer.InputConfig{Name: "companyProfile_input"},
	)

	companyProfileEmbedding := layer.NewEmbedding(
		companyProfileProcessor.Tokenizer().NumWords()+1,
		32,
		layer.EmbeddingConfig{Name: "company_profile_embedding"},
	)(companyProfileInput)

	companyProfileLSTM := layer.NewLSTM(32, layer.LSTMConfig{Name: "company_profile_lstm"})(companyProfileEmbedding)

	descriptionInput := layer.NewInput(
		tf.MakeShape(-1, int64(descriptionProcessor.Tokenizer().MaxLen())),
		layer.Float32,
		layer.InputConfig{Name: "description_input"},
	)

	descriptionEmbedding := layer.NewEmbedding(
		descriptionProcessor.Tokenizer().NumWords()+1,
		32,
		layer.EmbeddingConfig{Name: "description_embedding"},
	)(descriptionInput)

	descriptionLSTM := layer.NewLSTM(32, layer.LSTMConfig{Name: "description_lstm"})(descriptionEmbedding)

	requirementsInput := layer.NewInput(
		tf.MakeShape(-1, int64(requirementsProcessor.Tokenizer().MaxLen())),
		layer.Float32,
		layer.InputConfig{Name: "requirements_input"},
	)

	requirementsEmbedding := layer.NewEmbedding(
		requirementsProcessor.Tokenizer().NumWords()+1,
		32,
		layer.EmbeddingConfig{Name: "requirements_embedding"},
	)(requirementsInput)

	requirementsLSTM := layer.NewLSTM(32, layer.LSTMConfig{Name: "requirements_lstm"})(requirementsEmbedding)

	// Merge our LSTM layers into a single tensor
	concatenate := layer.NewConcatenate(-1)(titleLSTM, locationLSTM, departmentLSTM, companyProfileLSTM, descriptionLSTM, requirementsLSTM)

	// Feed the merged input into a dense network
	mergedDense1 := layer.NewDense(
		100,
		layer.Float32,
		layer.DenseConfig{Name: "merged_dense_1", Activation: "swish"},
	)(concatenate)
	mergedDense2 := layer.NewDense(
		100,
		layer.Float32,
		layer.DenseConfig{Name: "merged_dense_2", Activation: "swish"},
	)(mergedDense1)

	// Get the number of classes from the dataset if we don't want to count them manually, but in this case it is only 2
	output := layer.NewDense(
		dataset.NumCategoricalClasses(),
		layer.Float32,
		layer.DenseConfig{Name: "output", Activation: "softmax"},
	)(mergedDense2)

	// Define a keras style Functional model
	// Note that you don't need to pass in the inputs, the output variable contains all the other nodes as long as you use the same syntax of layer.New()(input)
	m := model.NewModel(
		logger,
		errorHandler,
		output,
	)

	// This part is pretty nasty under the hood. Effectively it will generate some python code for our model and execute it to save the model in a format we can load and train
	// A python binary must be available to use for this to work
	// The batchSize MUST match the batch size in the call to Fit or Evaluate
	batchSize := 200
	e = m.CompileAndLoad(batchSize)
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
			2021-12-11 12:15:29.145 : single_file_dataset.go:166 : Reading line offsets and counting stats
			2021-12-11 12:15:29.233 : single_file_dataset.go:267 : Found 17879 rows. Got class counts: map[int]int{0:17013, 1:866}
			2021-12-11 12:15:29.233 : single_file_dataset.go:298 : Fitting Pre-Processors
			Fitting preprocessors: 17476 17476/s2021-12-11 12:15:30.025 : single_file_dataset.go:155 : Loading line offsets and stats from cache file
			2021-12-11 12:15:30.025 : single_file_dataset.go:161 : Found 17879 rows. Got class counts: map[int]int{0:17013, 1:866}
			2021-12-11 12:15:30.025 : single_file_dataset.go:398 : Fit tokenizers
			2021-12-11 12:15:30.026 : main.go:168 : Shuffling dataset
			2021-12-11 12:15:30.027 : model.go:715 : Compiling and loading model. If anything goes wrong python error messages will be printed out.
			Initialising model
			Tracing learn
			Tracing evaluate
			Tracing predict
			Saving model
			Completed model base
			2021-12-11 12:15:57.609 : main.go:294 : Training model: examples/jobs/saved_models/trained_model
			2021-12-11 12:16:11.290 : logger.go:102 : End 1 8/8 (10s/10s) loss: 0.0650 acc: 0.7875 val_loss: 0.0729 val_acc: 0.9537
			2021-12-11 12:16:11.713 : logger.go:79 : Saved
			2021-12-11 12:16:20.775 : logger.go:102 : End 2 8/8 (9s/9s) loss: 0.0339 acc: 0.8660 val_loss: 0.0277 val_acc: 0.9156
			2021-12-11 12:16:29.844 : logger.go:102 : End 3 8/8 (9s/9s) loss: 0.0170 acc: 0.9183 val_loss: 0.0288 val_acc: 0.9444
			2021-12-11 12:16:38.930 : logger.go:102 : End 4 8/8 (9s/9s) loss: 0.0097 acc: 0.9541 val_loss: 0.0331 val_acc: 0.9425
			2021-12-11 12:16:47.940 : logger.go:102 : End 5 8/8 (8s/8s) loss: 0.0071 acc: 0.9661 val_loss: 0.0329 val_acc: 0.9150
			2021-12-11 12:16:57.036 : logger.go:102 : End 6 8/8 (9s/9s) loss: 0.0059 acc: 0.9661 val_loss: 0.0711 val_acc: 0.9675
			2021-12-11 12:16:57.044 : logger.go:79 : Saved
			2021-12-11 12:17:06.176 : logger.go:102 : End 7 8/8 (9s/9s) loss: 0.0057 acc: 0.9752 val_loss: 0.0643 val_acc: 0.9550
			2021-12-11 12:17:15.332 : logger.go:102 : End 8 8/8 (9s/9s) loss: 0.0036 acc: 0.9819 val_loss: 0.0736 val_acc: 0.9462
			2021-12-11 12:17:24.489 : logger.go:102 : End 9 8/8 (9s/9s) loss: 0.0046 acc: 0.9754 val_loss: 0.0471 val_acc: 0.9250
			2021-12-11 12:17:33.080 : logger.go : Train 10 68/71 (9s/9s) loss: 0.0041 acc: 0.9755 | Prefetched 32021-12-11 12:17:33.629857: I tensorflow/cc/saved_model/reader.cc:38] Reading SavedModel from: examples/jobs/saved_models/trained_model
			2021-12-11 12:17:33.629 : logger.go:102 : End 10 8/8 (9s/9s) loss: 0.0042 acc: 0.9753 val_loss: 0.0551 val_acc: 0.9244
			2021-12-11 12:17:33.629 : main.go:336 : Finished training
			2021-12-11 12:17:35.103 : inference.go:26 : Initialising inference provider with processors loaded from: examples/jobs/saved_models/trained_model
			2021-12-11 12:17:36.200 : main.go:435 : Predicted classes: [1 7.840193e-11]
	*/
}

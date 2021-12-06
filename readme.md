# TFKG - A Tensorflow and Keras Golang port

## This is experimental and quite nasty under the hood*

## Support
- macOS: running docker container, no GPU acceleration
- Ubuntu 18.04: binary execution on linux ubuntu with CUDA 11.2 and Python 3.8, with GPU acceleration
- Windows: ?

## Find your version
Versions starting with v0 are liable to change radically.
- Tensorflow 2.6 experimental support: `go get github.com/codingbeard/tfkg v0.2.6`

## Requirements
- Docker if using the provided container
- For GPU support in the container see: https://www.tensorflow.org/install/docker

**If not using the container: Make sure to install the correct versions to match the version of this library**
- Tensorflow C library: https://www.tensorflow.org/install/lang_c
- Python 3.8 - the binary "python" must be on your path and the correct version
- Tensorflow Python library: https://www.tensorflow.org/install

## Features
- Nvidia CUDA support on applicable platforms during Golang training/evaluation due to using the Tensorflow C library
- Define, Train, evaluate, and save Tensorflow compatible models in Golang
- Load models created with this library in Golang
- Load, shuffle, and preprocess csv datasets efficiently, even very large ones (tested on 330GB csv file)
    - String Tokenizer
    - Float/Int normalization to between 0-1

## Keras model types supported
- `tensorflow.keras.Sequential`
- Functional coming soon

## Keras Layers supported
- `tensorflow.keras.layers.Input`
- `tensorflow.keras.layers.Dense`
- More coming soon
- Interfaced - you can define custom layers

## Metrics
- Sparse Categorical Cross Entropy:
  - Accuracy
  - False positive rate at true positive rate (Specificity at Sensitivity)
  - True positive rate at false positive rate (Sensitivity at Specificity)

## Limitations
- Python Tensorflow Libraries are still required to use this library, though the docker container has it all
- This is an incomplete port of Tensorflow/Keras: There are many layers, metrics, and optimisers not yet ported
- There is no community support or documentation. You must understand Tensorflow/Keras and Golang well to have a chance of getting this working on a new project
- Loading/Training models saved by vanilla python Tensorflow/Keras is not supported, and may never be
- Image datasets and preprocessing is not yet supported
- Class weighting only works for datasets with two categories currently: negative/positive. It /will/ cause unintended side effects if you have more than two categories with imbalanced classes

## Examples:
See the full example with comments in ./examples/iris

To test it out run:
```
docker-compose up -d
make examples-iris
```

Define a model:
```go
m := model.NewSequentialModel(
    errorHandler,
    logger,
    layer.NewInput(tf.MakeShape(-1, 4), layer.Float32, layer.InputConfig{Name: "petal_sizes"}),
    layer.NewDense(100, layer.Float32, layer.DenseConfig{Activation: "swish"}),
    layer.NewDense(100, layer.Float32, layer.DenseConfig{Activation: "swish"}),
    layer.NewDense(3, layer.Float32, layer.DenseConfig{Activation: "softmax"}),
)
e = m.CompileAndLoad(
    3, // batchSize
)
```
Load a dataset:
```go
dataset, e := data.NewSingleFileDataset(
    logger,
    errorHandler,
    "examples/iris/data/iris.data", // filePath
    cacheDir,
    4, // categoryOffset
    0.8, // trainPercentage
    0.1, // valPercentage
    0.1, // testPercentage
    preprocessor.NewProcessor(
        errorHandler,
        cacheDir,
        "petal_sizes", // name
        0, // offset
        4, // dataLength
        true, // requiresFit
        preprocessor.NewDivisor(errorHandler),
        nil, // tokenizer
        preprocessor.ReadCsvFloat32s,
        preprocessor.ConvertDivisorToFloat32SliceTensor,
    ),
)
```
Train a model:
```go
m.Fit(
    "learn", // trainSignature
    "evaluate", // evaluateSignature
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
```


## *Nasty under the hood

The Tensorflow/Keras python package saves a Graph (see more: https://www.tensorflow.org/guide/intro_to_graphs) which can be executed in other languages using their C library as long as there are C bindings. 

The C library does not contain all the functionality of the python library when it comes to defining and saving models, it can only execute Graphs. 

The Graph is calculated in python based on your model configuration, and a lot of clever code on the part of the developers in optimising the graph. 

While possible, it is not currently feasible for me to generate the Graph in Golang, so I am relying on python to do so.

This means while the model is technically defined and trained in Golang, it just generates a json config string which static python code uses to configure the model and then saves it ready for loading in Golang for training. For the moment this is a needed evil.

If some kind soul wants to replicate Keras and Autograph to generate the Graph in Golang, feel free to make a pull request. I may eventually do it, but it is not likely. There is a branch origin/scratch which allows you to investigate the graph of a saved model.


## Acknowledgements

Big shout out to github.com/galeone for their Tensorflow Golang fork for 2.6 and again for their article on how to train a model in golang which helped me figure out how to then save the trained variables: https://pgaleone.eu/tensorflow/go/2020/11/27/deploy-train-tesorflow-models-in-go-human-activity-recognition/
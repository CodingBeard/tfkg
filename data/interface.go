package data

import (
	"errors"
	tf "github.com/galeone/tensorflow/tensorflow/go"
)

type GeneratorMode string

var (
	ErrGeneratorEnd                  = errors.New("end of data")
	GeneratorModeTrain GeneratorMode = "train"
	GeneratorModeVal   GeneratorMode = "val"
	GeneratorModeTest  GeneratorMode = "test"
)

type Dataset interface {
	Len() int
	NumCategoricalClasses() int
	SetMode(mode GeneratorMode) Dataset
	Shuffle(seed int64)
	Unshuffle() error
	GetColumnNames() []string
	GeneratorChan(batchSize int, preFetch int) chan Batch
	Generate(batchSize int) ([]*tf.Tensor, *tf.Tensor, *tf.Tensor, error)
	Reset() error
	SaveProcessors(saveDir string) error
}

type Batch struct {
	X            []*tf.Tensor
	Y            *tf.Tensor
	ClassWeights *tf.Tensor
}

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
	SetMode(mode GeneratorMode) Dataset
	Shuffle(seed int64)
	Unshuffle() error
	GetColumnNames() []string
	GeneratorChan(batchSize int, preFetch int) chan Batch
	Generate(batchSize int) ([]*tf.Tensor, *tf.Tensor, error)
	Reset() error
}

type Batch struct {
	X []*tf.Tensor
	Y *tf.Tensor
	// TODO: change class weights to a single tensor
	PosWeight *tf.Tensor
	NegWeight *tf.Tensor
}

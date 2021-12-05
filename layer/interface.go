package layer

import (
	tf "github.com/codingbeard/tfkg/tensorflow/go"
	"github.com/codingbeard/tfkg/tensorflow/go/op"
)

type Layer interface {
	Errors() []error
	Compile(scope *op.Scope) []error
	Shape() tf.Shape
	Output() tf.Output
}

type AcceptsInput interface {
	SetInput(input Layer)
}

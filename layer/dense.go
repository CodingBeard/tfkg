package layer

import (
	"fmt"
	tf "github.com/codingbeard/tfkg/tensorflow/go"
	"github.com/codingbeard/tfkg/tensorflow/go/op"
	"github.com/codingbeard/tfkg/util"
	"math/rand"
)

type Dense struct {
	units    int
	input    Layer
	outputOp tf.Output
	shape    tf.Shape

	config DenseConfig
	es     []error
}

type DenseConfig struct {
	Name  string
	Dtype tf.DataType
}

func NewDense(units int, config DenseConfig) func(input Layer) *Dense {
	var es []error
	if config.Name == "" {
		config.Name = util.UniqueName("dense")
	}
	if !util.IsValidDtype(config.Dtype) {
		es = append(es, util.NewError(fmt.Errorf("no dtype specified for Dense %s", config.Name)))
	}
	return func(input Layer) *Dense {
		return &Dense{
			units:  units,
			input:  input,
			shape:  tf.MakeShape(-1, int64(units)),
			config: config,
			es:     es,
		}
	}
}

func (i *Dense) SetInput(input Layer) {
	i.input = input
}

func (i *Dense) Errors() []error {
	return i.es
}

func (i *Dense) Compile(scope *op.Scope) []error {
	if i.input == nil {
		i.es = append(i.es, util.NewError(fmt.Errorf(
			"dense layer %s has no input assigned to it",
			i.config.Name,
		)))
	}
	if len(i.Errors()) != 0 {
		return i.Errors()
	}

	scope = scope.SubScope(i.config.Name).WithControlDependencies(i.input.Output().Op)

	weightsHandleOp := op.VarHandleOp(scope.SubScope("handle"), tf.Float, tf.MakeShape(int64(i.units), i.input.Shape().Size(1)))

	var weights [][]float32
	for j := 0; j < i.units; j++ {
		var subWeights []float32
		for k := 0; k < int(i.input.Shape().Size(1)); k++ {
			subWeights = append(subWeights, rand.Float32())
		}
		weights = append(weights, subWeights)
	}

	defaultWeights := op.Const(scope.SubScope("weights"), weights)

	scope = scope.WithControlDependencies(
		op.AssignVariableOp(scope.SubScope("assign"), weightsHandleOp, defaultWeights),
	)

	aReadOp := op.ReadVariableOp(scope.SubScope("read"), weightsHandleOp, tf.Float)

	castOp := op.Cast(scope.SubScope("cast"), i.input.Output(), tf.Float)

	i.outputOp = op.MatMul(scope.SubScope("matmul"), castOp, aReadOp, op.MatMulTransposeB(true))

	return nil
}

func (i *Dense) Output() tf.Output {
	return i.outputOp
}

func (i *Dense) Shape() tf.Shape {
	return i.shape
}

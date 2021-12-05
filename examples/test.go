package main

import (
	"fmt"
	"github.com/codingbeard/tfkg/layer"
	"github.com/codingbeard/tfkg/model"
	tf "github.com/codingbeard/tfkg/tensorflow/go"
	"github.com/codingbeard/tfkg/tensorflow/go/op"
	"io/ioutil"
	"os"
)

func main() {
	//mul()
	//input()
	//dense()
	//exploreModel()
	decompile()
}

func decompile() {
	m, e := model.Load("base_model", []string{"serve"}, nil)
	if e != nil {
		panic(e)
	}

	decompiled, e := m.DecompileGraphToGolangCode()
	if e != nil {
		panic(e)
	}
	fmt.Println(decompiled)
}

func exploreModel() {
	graph := tf.NewGraph()
	fileBytes, e := ioutil.ReadFile("base_model/graph_def_predict.graph")
	if e != nil {
		panic(e)
	}
	e = graph.Import(fileBytes, "")
	if e != nil {
		panic(e)
	}

	e = ioutil.WriteFile("base_model/graph_def_predict.txt", []byte(graph.GetDebugString()), os.ModePerm)
	if e != nil {
		panic(e)
	}

	graph = tf.NewGraph()
	fileBytes, e = ioutil.ReadFile("base_model/graph_def_learn.graph")
	if e != nil {
		panic(e)
	}
	e = graph.Import(fileBytes, "")
	if e != nil {
		panic(e)
	}

	e = ioutil.WriteFile("base_model/graph_def_learn.txt", []byte(graph.GetDebugString()), os.ModePerm)
	if e != nil {
		panic(e)
	}

	m, e := tf.LoadSavedModel("base_model", []string{"serve"}, nil)
	if e != nil {
		panic(e)
	}

	e = ioutil.WriteFile("base_model/saved_model_graph.txt", []byte(m.Graph.GetDebugString()), os.ModePerm)
	if e != nil {
		panic(e)
	}

}

func dense() {
	scope := op.NewScope()
	input := layer.NewInput(layer.InputConfig{
		Dtype: tf.Int32,
		Shape: tf.MakeShape(-1, 5),
	})

	dense := layer.NewDense(2, layer.DenseConfig{
		Dtype: tf.Int32,
	})(input)

	dense2 := layer.NewDense(1, layer.DenseConfig{
		Dtype: tf.Int32,
	})(dense)

	es := input.Compile(scope)
	if len(es) > 0 {
		for _, e := range es {
			fmt.Println(e)
		}
		return
	}
	es = dense.Compile(scope)
	if len(es) > 0 {
		for _, e := range es {
			fmt.Println(e)
		}
		return
	}
	es = dense2.Compile(scope)
	if len(es) > 0 {
		for _, e := range es {
			fmt.Println(e)
		}
		return
	}

	graph, e := scope.Finalize()
	if e != nil {
		panic(e)
	}

	for _, operation := range graph.Operations() {
		fmt.Println(operation.Name(), operation.Type())
	}

	session, e := tf.NewSession(graph, nil)
	if e != nil {
		panic(e)
	}

	inputTensor, e := tf.NewTensor([][]int32{{1, 2, 3, 4, 5}})
	if e != nil {
		panic(e)
	}

	result, e := session.Run(
		map[tf.Output]*tf.Tensor{
			input.Output(): inputTensor,
		},
		[]tf.Output{
			dense2.Output(),
		},
		nil,
	)
	if e != nil {
		panic(e)
	}

	fmt.Println(result[0].Value())
}

func input() {
	scope := op.NewScope()
	input := layer.NewInput(layer.InputConfig{
		Dtype: tf.Int32,
		Shape: tf.MakeShape(1),
	})
	es := input.Compile(scope)
	if len(es) > 0 {
		for _, e := range es {
			fmt.Println(e)
		}
		return
	}

	aHandleOp := op.VarHandleOp(scope.SubScope("handle"), tf.Int32, tf.MakeShape(1))

	scope = scope.WithControlDependencies(
		op.AssignVariableOp(scope.SubScope("assign"), aHandleOp, input.Output()),
	)

	aReadOp := op.ReadVariableOp(scope.SubScope("read"), aHandleOp, tf.Int32)

	graph, e := scope.Finalize()
	if e != nil {
		panic(e)
	}

	session, e := tf.NewSession(graph, nil)
	if e != nil {
		panic(e)
	}

	inputTensor, e := tf.NewTensor([]int32{3})
	if e != nil {
		panic(e)
	}

	result, e := session.Run(
		map[tf.Output]*tf.Tensor{
			input.Output(): inputTensor,
		},
		[]tf.Output{
			aReadOp,
		},
		nil,
	)
	if e != nil {
		panic(e)
	}

	fmt.Println(result[0].Value())
}

func mul() {
	scope := op.NewScope()
	aOp := op.Const(scope.SubScope("const"), [1]int32{})
	bOp := op.Const(scope.SubScope("const"), [1]int32{})

	aHandleOp := op.VarHandleOp(scope.SubScope("handle"), tf.Int32, tf.MakeShape(1))
	bHandleOp := op.VarHandleOp(scope.SubScope("handle"), tf.Int32, tf.MakeShape(1))

	scope = scope.WithControlDependencies(
		op.AssignVariableOp(scope.SubScope("assign"), aHandleOp, aOp),
		op.AssignVariableOp(scope.SubScope("assign"), bHandleOp, bOp),
	)

	aReadOp := op.ReadVariableOp(scope.SubScope("read"), aHandleOp, tf.Int32)
	bReadOp := op.ReadVariableOp(scope.SubScope("read"), bHandleOp, tf.Int32)

	y := op.Mul(scope, aReadOp, bReadOp)

	g, e := scope.Finalize()
	if e != nil {
		panic(e)
	}

	s, e := tf.NewSession(g, nil)
	if e != nil {
		panic(e)
	}

	a, e := tf.NewTensor([]int32{3})
	if e != nil {
		panic(e)
	}
	b, e := tf.NewTensor([]int32{3})
	if e != nil {
		panic(e)
	}

	result, e := s.Run(
		map[tf.Output]*tf.Tensor{
			aOp: a,
			bOp: b,
		},
		[]tf.Output{
			y,
		},
		nil,
	)

	fmt.Println(result[0].Value())
}

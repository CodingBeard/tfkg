package layer

import "C"
import (
	"fmt"
	tf "github.com/galeone/tensorflow/tensorflow/go"
)

type TfBool string

var (
	TfTrue    TfBool = "True"
	TfFalse   TfBool = "False"
	TfDefault TfBool = "None"
)

type DataType string

// Types of scalar values in the TensorFlow type system.
const (
	Float16    DataType = "tf.float16"
	Float32    DataType = "tf.float32"
	Float64    DataType = "tf.float64"
	Double     DataType = "tf.double"
	Int32      DataType = "tf.int32"
	Uint32     DataType = "tf.uint32"
	Uint8      DataType = "tf.uint8"
	Int16      DataType = "tf.int16"
	Int8       DataType = "tf.int8"
	String     DataType = "tf.string"
	Complex64  DataType = "tf.complex64"
	Complex    DataType = "tf.complex"
	Int64      DataType = "tf.int64"
	Uint64     DataType = "tf.uint64"
	Bool       DataType = "tf.bool"
	Qint8      DataType = "tf.qint8"
	Quint8     DataType = "tf.quint8"
	Qint32     DataType = "tf.qint32"
	Bfloat16   DataType = "tf.bfloat16"
	Qint16     DataType = "tf.qint16"
	Quint16    DataType = "tf.quint16"
	Uint16     DataType = "tf.uint16"
	Complex128 DataType = "tf.complex128"
	Half       DataType = "tf.half"
)

type Layer interface {
	GetShape() tf.Shape
	GetDtype() DataType
	SetInput(inputs []Layer)
	GetImport() string
	GetPythonVariableName() string
	GetPythonDefinitionString() string
}

var uniqueNameCounts = make(map[string]int)

func uniqueName(name string) string {
	count := uniqueNameCounts[name]
	count++
	uniqueNameCounts[name] = count

	return fmt.Sprintf("%s_%d", name, count)
}

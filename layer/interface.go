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
	Float16    DataType = "float16"
	Float32    DataType = "float32"
	Float64    DataType = "float64"
	Double     DataType = "double"
	Int32      DataType = "int32"
	Uint32     DataType = "uint32"
	Uint8      DataType = "uint8"
	Int16      DataType = "int16"
	Int8       DataType = "int8"
	String     DataType = "string"
	Complex64  DataType = "complex64"
	Complex    DataType = "complex"
	Int64      DataType = "int64"
	Uint64     DataType = "uint64"
	Bool       DataType = "bool"
	Qint8      DataType = "qint8"
	Quint8     DataType = "quint8"
	Qint32     DataType = "qint32"
	Bfloat16   DataType = "bfloat16"
	Qint16     DataType = "qint16"
	Quint16    DataType = "quint16"
	Uint16     DataType = "uint16"
	Complex128 DataType = "complex128"
	Half       DataType = "half"
)

type Layer interface {
	GetShape() tf.Shape
	GetDtype() DataType
	SetInput(inputs []Layer)
	GetImport() string
	GetName() string
	GetKerasLayerConfig() interface{}
}

func (d TfBool) ToBool(defaultValue bool) bool {
	if d == TfTrue {
		return true
	} else if d == TfFalse {
		return false
	}
	return defaultValue
}

var uniqueNameCounts = make(map[string]int)

func uniqueName(name string) string {
	count := uniqueNameCounts[name]
	count++
	uniqueNameCounts[name] = count

	return fmt.Sprintf("%s_%d", name, count)
}

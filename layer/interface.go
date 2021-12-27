package layer

import "C"
import (
	"fmt"
	tf "github.com/galeone/tensorflow/tensorflow/go"
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
	SetInputs(inputs ...Layer) Layer
	GetInputs() []Layer
	GetName() string
	GetKerasLayerConfig() interface{}
	GetCustomLayerDefinition() string
}

var uniqueNameCounts = make(map[string]int)

func UniqueName(name string) string {
	count := uniqueNameCounts[name]
	count++
	uniqueNameCounts[name] = count

	return fmt.Sprintf("%s_%d", name, count)
}

func (d *DataType) String() string {
	return string(*d)
}

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.26.0
// 	protoc        v3.16.0
// source: tensorflow/core/framework/types.proto

package types_go_proto

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

// (== suppress_warning documentation-presence ==)
// LINT.IfChange
type DataType int32

const (
	// Not a legal value for DataType.  Used to indicate a DataType field
	// has not been set.
	DataType_DT_INVALID DataType = 0
	// Data types that all computation devices are expected to be
	// capable to support.
	DataType_DT_FLOAT      DataType = 1
	DataType_DT_DOUBLE     DataType = 2
	DataType_DT_INT32      DataType = 3
	DataType_DT_UINT8      DataType = 4
	DataType_DT_INT16      DataType = 5
	DataType_DT_INT8       DataType = 6
	DataType_DT_STRING     DataType = 7
	DataType_DT_COMPLEX64  DataType = 8 // Single-precision complex
	DataType_DT_INT64      DataType = 9
	DataType_DT_BOOL       DataType = 10
	DataType_DT_QINT8      DataType = 11 // Quantized int8
	DataType_DT_QUINT8     DataType = 12 // Quantized uint8
	DataType_DT_QINT32     DataType = 13 // Quantized int32
	DataType_DT_BFLOAT16   DataType = 14 // Float32 truncated to 16 bits.  Only for cast ops.
	DataType_DT_QINT16     DataType = 15 // Quantized int16
	DataType_DT_QUINT16    DataType = 16 // Quantized uint16
	DataType_DT_UINT16     DataType = 17
	DataType_DT_COMPLEX128 DataType = 18 // Double-precision complex
	DataType_DT_HALF       DataType = 19
	DataType_DT_RESOURCE   DataType = 20
	DataType_DT_VARIANT    DataType = 21 // Arbitrary C++ data types
	DataType_DT_UINT32     DataType = 22
	DataType_DT_UINT64     DataType = 23
	// Do not use!  These are only for parameters.  Every enum above
	// should have a corresponding value below (verified by types_test).
	DataType_DT_FLOAT_REF      DataType = 101
	DataType_DT_DOUBLE_REF     DataType = 102
	DataType_DT_INT32_REF      DataType = 103
	DataType_DT_UINT8_REF      DataType = 104
	DataType_DT_INT16_REF      DataType = 105
	DataType_DT_INT8_REF       DataType = 106
	DataType_DT_STRING_REF     DataType = 107
	DataType_DT_COMPLEX64_REF  DataType = 108
	DataType_DT_INT64_REF      DataType = 109
	DataType_DT_BOOL_REF       DataType = 110
	DataType_DT_QINT8_REF      DataType = 111
	DataType_DT_QUINT8_REF     DataType = 112
	DataType_DT_QINT32_REF     DataType = 113
	DataType_DT_BFLOAT16_REF   DataType = 114
	DataType_DT_QINT16_REF     DataType = 115
	DataType_DT_QUINT16_REF    DataType = 116
	DataType_DT_UINT16_REF     DataType = 117
	DataType_DT_COMPLEX128_REF DataType = 118
	DataType_DT_HALF_REF       DataType = 119
	DataType_DT_RESOURCE_REF   DataType = 120
	DataType_DT_VARIANT_REF    DataType = 121
	DataType_DT_UINT32_REF     DataType = 122
	DataType_DT_UINT64_REF     DataType = 123
)

// Enum value maps for DataType.
var (
	DataType_name = map[int32]string{
		0:   "DT_INVALID",
		1:   "DT_FLOAT",
		2:   "DT_DOUBLE",
		3:   "DT_INT32",
		4:   "DT_UINT8",
		5:   "DT_INT16",
		6:   "DT_INT8",
		7:   "DT_STRING",
		8:   "DT_COMPLEX64",
		9:   "DT_INT64",
		10:  "DT_BOOL",
		11:  "DT_QINT8",
		12:  "DT_QUINT8",
		13:  "DT_QINT32",
		14:  "DT_BFLOAT16",
		15:  "DT_QINT16",
		16:  "DT_QUINT16",
		17:  "DT_UINT16",
		18:  "DT_COMPLEX128",
		19:  "DT_HALF",
		20:  "DT_RESOURCE",
		21:  "DT_VARIANT",
		22:  "DT_UINT32",
		23:  "DT_UINT64",
		101: "DT_FLOAT_REF",
		102: "DT_DOUBLE_REF",
		103: "DT_INT32_REF",
		104: "DT_UINT8_REF",
		105: "DT_INT16_REF",
		106: "DT_INT8_REF",
		107: "DT_STRING_REF",
		108: "DT_COMPLEX64_REF",
		109: "DT_INT64_REF",
		110: "DT_BOOL_REF",
		111: "DT_QINT8_REF",
		112: "DT_QUINT8_REF",
		113: "DT_QINT32_REF",
		114: "DT_BFLOAT16_REF",
		115: "DT_QINT16_REF",
		116: "DT_QUINT16_REF",
		117: "DT_UINT16_REF",
		118: "DT_COMPLEX128_REF",
		119: "DT_HALF_REF",
		120: "DT_RESOURCE_REF",
		121: "DT_VARIANT_REF",
		122: "DT_UINT32_REF",
		123: "DT_UINT64_REF",
	}
	DataType_value = map[string]int32{
		"DT_INVALID":        0,
		"DT_FLOAT":          1,
		"DT_DOUBLE":         2,
		"DT_INT32":          3,
		"DT_UINT8":          4,
		"DT_INT16":          5,
		"DT_INT8":           6,
		"DT_STRING":         7,
		"DT_COMPLEX64":      8,
		"DT_INT64":          9,
		"DT_BOOL":           10,
		"DT_QINT8":          11,
		"DT_QUINT8":         12,
		"DT_QINT32":         13,
		"DT_BFLOAT16":       14,
		"DT_QINT16":         15,
		"DT_QUINT16":        16,
		"DT_UINT16":         17,
		"DT_COMPLEX128":     18,
		"DT_HALF":           19,
		"DT_RESOURCE":       20,
		"DT_VARIANT":        21,
		"DT_UINT32":         22,
		"DT_UINT64":         23,
		"DT_FLOAT_REF":      101,
		"DT_DOUBLE_REF":     102,
		"DT_INT32_REF":      103,
		"DT_UINT8_REF":      104,
		"DT_INT16_REF":      105,
		"DT_INT8_REF":       106,
		"DT_STRING_REF":     107,
		"DT_COMPLEX64_REF":  108,
		"DT_INT64_REF":      109,
		"DT_BOOL_REF":       110,
		"DT_QINT8_REF":      111,
		"DT_QUINT8_REF":     112,
		"DT_QINT32_REF":     113,
		"DT_BFLOAT16_REF":   114,
		"DT_QINT16_REF":     115,
		"DT_QUINT16_REF":    116,
		"DT_UINT16_REF":     117,
		"DT_COMPLEX128_REF": 118,
		"DT_HALF_REF":       119,
		"DT_RESOURCE_REF":   120,
		"DT_VARIANT_REF":    121,
		"DT_UINT32_REF":     122,
		"DT_UINT64_REF":     123,
	}
)

func (x DataType) Enum() *DataType {
	p := new(DataType)
	*p = x
	return p
}

func (x DataType) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (DataType) Descriptor() protoreflect.EnumDescriptor {
	return file_tensorflow_core_framework_types_proto_enumTypes[0].Descriptor()
}

func (DataType) Type() protoreflect.EnumType {
	return &file_tensorflow_core_framework_types_proto_enumTypes[0]
}

func (x DataType) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use DataType.Descriptor instead.
func (DataType) EnumDescriptor() ([]byte, []int) {
	return file_tensorflow_core_framework_types_proto_rawDescGZIP(), []int{0}
}

// For identifying the underlying type of a variant. For variants, the types
// listed here are a subset of the types in the variant type registry,
// corresponding to commonly used variants which must occasionally be
// special-cased.
type SpecializedType int32

const (
	// Invalid/unknown specialized type.
	SpecializedType_ST_INVALID SpecializedType = 0
	// "tensorflow::TensorList" in the variant type registry.
	SpecializedType_ST_TENSOR_LIST SpecializedType = 1
	// "tensorflow::data::Optional" in the variant type registry.
	SpecializedType_ST_OPTIONAL SpecializedType = 2
)

// Enum value maps for SpecializedType.
var (
	SpecializedType_name = map[int32]string{
		0: "ST_INVALID",
		1: "ST_TENSOR_LIST",
		2: "ST_OPTIONAL",
	}
	SpecializedType_value = map[string]int32{
		"ST_INVALID":     0,
		"ST_TENSOR_LIST": 1,
		"ST_OPTIONAL":    2,
	}
)

func (x SpecializedType) Enum() *SpecializedType {
	p := new(SpecializedType)
	*p = x
	return p
}

func (x SpecializedType) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (SpecializedType) Descriptor() protoreflect.EnumDescriptor {
	return file_tensorflow_core_framework_types_proto_enumTypes[1].Descriptor()
}

func (SpecializedType) Type() protoreflect.EnumType {
	return &file_tensorflow_core_framework_types_proto_enumTypes[1]
}

func (x SpecializedType) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use SpecializedType.Descriptor instead.
func (SpecializedType) EnumDescriptor() ([]byte, []int) {
	return file_tensorflow_core_framework_types_proto_rawDescGZIP(), []int{1}
}

var File_tensorflow_core_framework_types_proto protoreflect.FileDescriptor

var file_tensorflow_core_framework_types_proto_rawDesc = []byte{
	0x0a, 0x25, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2f, 0x63, 0x6f, 0x72,
	0x65, 0x2f, 0x66, 0x72, 0x61, 0x6d, 0x65, 0x77, 0x6f, 0x72, 0x6b, 0x2f, 0x74, 0x79, 0x70, 0x65,
	0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x0a, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66,
	0x6c, 0x6f, 0x77, 0x2a, 0xaa, 0x06, 0x0a, 0x08, 0x44, 0x61, 0x74, 0x61, 0x54, 0x79, 0x70, 0x65,
	0x12, 0x0e, 0x0a, 0x0a, 0x44, 0x54, 0x5f, 0x49, 0x4e, 0x56, 0x41, 0x4c, 0x49, 0x44, 0x10, 0x00,
	0x12, 0x0c, 0x0a, 0x08, 0x44, 0x54, 0x5f, 0x46, 0x4c, 0x4f, 0x41, 0x54, 0x10, 0x01, 0x12, 0x0d,
	0x0a, 0x09, 0x44, 0x54, 0x5f, 0x44, 0x4f, 0x55, 0x42, 0x4c, 0x45, 0x10, 0x02, 0x12, 0x0c, 0x0a,
	0x08, 0x44, 0x54, 0x5f, 0x49, 0x4e, 0x54, 0x33, 0x32, 0x10, 0x03, 0x12, 0x0c, 0x0a, 0x08, 0x44,
	0x54, 0x5f, 0x55, 0x49, 0x4e, 0x54, 0x38, 0x10, 0x04, 0x12, 0x0c, 0x0a, 0x08, 0x44, 0x54, 0x5f,
	0x49, 0x4e, 0x54, 0x31, 0x36, 0x10, 0x05, 0x12, 0x0b, 0x0a, 0x07, 0x44, 0x54, 0x5f, 0x49, 0x4e,
	0x54, 0x38, 0x10, 0x06, 0x12, 0x0d, 0x0a, 0x09, 0x44, 0x54, 0x5f, 0x53, 0x54, 0x52, 0x49, 0x4e,
	0x47, 0x10, 0x07, 0x12, 0x10, 0x0a, 0x0c, 0x44, 0x54, 0x5f, 0x43, 0x4f, 0x4d, 0x50, 0x4c, 0x45,
	0x58, 0x36, 0x34, 0x10, 0x08, 0x12, 0x0c, 0x0a, 0x08, 0x44, 0x54, 0x5f, 0x49, 0x4e, 0x54, 0x36,
	0x34, 0x10, 0x09, 0x12, 0x0b, 0x0a, 0x07, 0x44, 0x54, 0x5f, 0x42, 0x4f, 0x4f, 0x4c, 0x10, 0x0a,
	0x12, 0x0c, 0x0a, 0x08, 0x44, 0x54, 0x5f, 0x51, 0x49, 0x4e, 0x54, 0x38, 0x10, 0x0b, 0x12, 0x0d,
	0x0a, 0x09, 0x44, 0x54, 0x5f, 0x51, 0x55, 0x49, 0x4e, 0x54, 0x38, 0x10, 0x0c, 0x12, 0x0d, 0x0a,
	0x09, 0x44, 0x54, 0x5f, 0x51, 0x49, 0x4e, 0x54, 0x33, 0x32, 0x10, 0x0d, 0x12, 0x0f, 0x0a, 0x0b,
	0x44, 0x54, 0x5f, 0x42, 0x46, 0x4c, 0x4f, 0x41, 0x54, 0x31, 0x36, 0x10, 0x0e, 0x12, 0x0d, 0x0a,
	0x09, 0x44, 0x54, 0x5f, 0x51, 0x49, 0x4e, 0x54, 0x31, 0x36, 0x10, 0x0f, 0x12, 0x0e, 0x0a, 0x0a,
	0x44, 0x54, 0x5f, 0x51, 0x55, 0x49, 0x4e, 0x54, 0x31, 0x36, 0x10, 0x10, 0x12, 0x0d, 0x0a, 0x09,
	0x44, 0x54, 0x5f, 0x55, 0x49, 0x4e, 0x54, 0x31, 0x36, 0x10, 0x11, 0x12, 0x11, 0x0a, 0x0d, 0x44,
	0x54, 0x5f, 0x43, 0x4f, 0x4d, 0x50, 0x4c, 0x45, 0x58, 0x31, 0x32, 0x38, 0x10, 0x12, 0x12, 0x0b,
	0x0a, 0x07, 0x44, 0x54, 0x5f, 0x48, 0x41, 0x4c, 0x46, 0x10, 0x13, 0x12, 0x0f, 0x0a, 0x0b, 0x44,
	0x54, 0x5f, 0x52, 0x45, 0x53, 0x4f, 0x55, 0x52, 0x43, 0x45, 0x10, 0x14, 0x12, 0x0e, 0x0a, 0x0a,
	0x44, 0x54, 0x5f, 0x56, 0x41, 0x52, 0x49, 0x41, 0x4e, 0x54, 0x10, 0x15, 0x12, 0x0d, 0x0a, 0x09,
	0x44, 0x54, 0x5f, 0x55, 0x49, 0x4e, 0x54, 0x33, 0x32, 0x10, 0x16, 0x12, 0x0d, 0x0a, 0x09, 0x44,
	0x54, 0x5f, 0x55, 0x49, 0x4e, 0x54, 0x36, 0x34, 0x10, 0x17, 0x12, 0x10, 0x0a, 0x0c, 0x44, 0x54,
	0x5f, 0x46, 0x4c, 0x4f, 0x41, 0x54, 0x5f, 0x52, 0x45, 0x46, 0x10, 0x65, 0x12, 0x11, 0x0a, 0x0d,
	0x44, 0x54, 0x5f, 0x44, 0x4f, 0x55, 0x42, 0x4c, 0x45, 0x5f, 0x52, 0x45, 0x46, 0x10, 0x66, 0x12,
	0x10, 0x0a, 0x0c, 0x44, 0x54, 0x5f, 0x49, 0x4e, 0x54, 0x33, 0x32, 0x5f, 0x52, 0x45, 0x46, 0x10,
	0x67, 0x12, 0x10, 0x0a, 0x0c, 0x44, 0x54, 0x5f, 0x55, 0x49, 0x4e, 0x54, 0x38, 0x5f, 0x52, 0x45,
	0x46, 0x10, 0x68, 0x12, 0x10, 0x0a, 0x0c, 0x44, 0x54, 0x5f, 0x49, 0x4e, 0x54, 0x31, 0x36, 0x5f,
	0x52, 0x45, 0x46, 0x10, 0x69, 0x12, 0x0f, 0x0a, 0x0b, 0x44, 0x54, 0x5f, 0x49, 0x4e, 0x54, 0x38,
	0x5f, 0x52, 0x45, 0x46, 0x10, 0x6a, 0x12, 0x11, 0x0a, 0x0d, 0x44, 0x54, 0x5f, 0x53, 0x54, 0x52,
	0x49, 0x4e, 0x47, 0x5f, 0x52, 0x45, 0x46, 0x10, 0x6b, 0x12, 0x14, 0x0a, 0x10, 0x44, 0x54, 0x5f,
	0x43, 0x4f, 0x4d, 0x50, 0x4c, 0x45, 0x58, 0x36, 0x34, 0x5f, 0x52, 0x45, 0x46, 0x10, 0x6c, 0x12,
	0x10, 0x0a, 0x0c, 0x44, 0x54, 0x5f, 0x49, 0x4e, 0x54, 0x36, 0x34, 0x5f, 0x52, 0x45, 0x46, 0x10,
	0x6d, 0x12, 0x0f, 0x0a, 0x0b, 0x44, 0x54, 0x5f, 0x42, 0x4f, 0x4f, 0x4c, 0x5f, 0x52, 0x45, 0x46,
	0x10, 0x6e, 0x12, 0x10, 0x0a, 0x0c, 0x44, 0x54, 0x5f, 0x51, 0x49, 0x4e, 0x54, 0x38, 0x5f, 0x52,
	0x45, 0x46, 0x10, 0x6f, 0x12, 0x11, 0x0a, 0x0d, 0x44, 0x54, 0x5f, 0x51, 0x55, 0x49, 0x4e, 0x54,
	0x38, 0x5f, 0x52, 0x45, 0x46, 0x10, 0x70, 0x12, 0x11, 0x0a, 0x0d, 0x44, 0x54, 0x5f, 0x51, 0x49,
	0x4e, 0x54, 0x33, 0x32, 0x5f, 0x52, 0x45, 0x46, 0x10, 0x71, 0x12, 0x13, 0x0a, 0x0f, 0x44, 0x54,
	0x5f, 0x42, 0x46, 0x4c, 0x4f, 0x41, 0x54, 0x31, 0x36, 0x5f, 0x52, 0x45, 0x46, 0x10, 0x72, 0x12,
	0x11, 0x0a, 0x0d, 0x44, 0x54, 0x5f, 0x51, 0x49, 0x4e, 0x54, 0x31, 0x36, 0x5f, 0x52, 0x45, 0x46,
	0x10, 0x73, 0x12, 0x12, 0x0a, 0x0e, 0x44, 0x54, 0x5f, 0x51, 0x55, 0x49, 0x4e, 0x54, 0x31, 0x36,
	0x5f, 0x52, 0x45, 0x46, 0x10, 0x74, 0x12, 0x11, 0x0a, 0x0d, 0x44, 0x54, 0x5f, 0x55, 0x49, 0x4e,
	0x54, 0x31, 0x36, 0x5f, 0x52, 0x45, 0x46, 0x10, 0x75, 0x12, 0x15, 0x0a, 0x11, 0x44, 0x54, 0x5f,
	0x43, 0x4f, 0x4d, 0x50, 0x4c, 0x45, 0x58, 0x31, 0x32, 0x38, 0x5f, 0x52, 0x45, 0x46, 0x10, 0x76,
	0x12, 0x0f, 0x0a, 0x0b, 0x44, 0x54, 0x5f, 0x48, 0x41, 0x4c, 0x46, 0x5f, 0x52, 0x45, 0x46, 0x10,
	0x77, 0x12, 0x13, 0x0a, 0x0f, 0x44, 0x54, 0x5f, 0x52, 0x45, 0x53, 0x4f, 0x55, 0x52, 0x43, 0x45,
	0x5f, 0x52, 0x45, 0x46, 0x10, 0x78, 0x12, 0x12, 0x0a, 0x0e, 0x44, 0x54, 0x5f, 0x56, 0x41, 0x52,
	0x49, 0x41, 0x4e, 0x54, 0x5f, 0x52, 0x45, 0x46, 0x10, 0x79, 0x12, 0x11, 0x0a, 0x0d, 0x44, 0x54,
	0x5f, 0x55, 0x49, 0x4e, 0x54, 0x33, 0x32, 0x5f, 0x52, 0x45, 0x46, 0x10, 0x7a, 0x12, 0x11, 0x0a,
	0x0d, 0x44, 0x54, 0x5f, 0x55, 0x49, 0x4e, 0x54, 0x36, 0x34, 0x5f, 0x52, 0x45, 0x46, 0x10, 0x7b,
	0x2a, 0x46, 0x0a, 0x0f, 0x53, 0x70, 0x65, 0x63, 0x69, 0x61, 0x6c, 0x69, 0x7a, 0x65, 0x64, 0x54,
	0x79, 0x70, 0x65, 0x12, 0x0e, 0x0a, 0x0a, 0x53, 0x54, 0x5f, 0x49, 0x4e, 0x56, 0x41, 0x4c, 0x49,
	0x44, 0x10, 0x00, 0x12, 0x12, 0x0a, 0x0e, 0x53, 0x54, 0x5f, 0x54, 0x45, 0x4e, 0x53, 0x4f, 0x52,
	0x5f, 0x4c, 0x49, 0x53, 0x54, 0x10, 0x01, 0x12, 0x0f, 0x0a, 0x0b, 0x53, 0x54, 0x5f, 0x4f, 0x50,
	0x54, 0x49, 0x4f, 0x4e, 0x41, 0x4c, 0x10, 0x02, 0x42, 0x7a, 0x0a, 0x18, 0x6f, 0x72, 0x67, 0x2e,
	0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2e, 0x66, 0x72, 0x61, 0x6d, 0x65,
	0x77, 0x6f, 0x72, 0x6b, 0x42, 0x0b, 0x54, 0x79, 0x70, 0x65, 0x73, 0x50, 0x72, 0x6f, 0x74, 0x6f,
	0x73, 0x50, 0x01, 0x5a, 0x4c, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f,
	0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2f, 0x74, 0x65, 0x6e, 0x73, 0x6f,
	0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2f, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f, 0x77,
	0x2f, 0x67, 0x6f, 0x2f, 0x63, 0x6f, 0x72, 0x65, 0x2f, 0x66, 0x72, 0x61, 0x6d, 0x65, 0x77, 0x6f,
	0x72, 0x6b, 0x2f, 0x74, 0x79, 0x70, 0x65, 0x73, 0x5f, 0x67, 0x6f, 0x5f, 0x70, 0x72, 0x6f, 0x74,
	0x6f, 0xf8, 0x01, 0x01, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_tensorflow_core_framework_types_proto_rawDescOnce sync.Once
	file_tensorflow_core_framework_types_proto_rawDescData = file_tensorflow_core_framework_types_proto_rawDesc
)

func file_tensorflow_core_framework_types_proto_rawDescGZIP() []byte {
	file_tensorflow_core_framework_types_proto_rawDescOnce.Do(func() {
		file_tensorflow_core_framework_types_proto_rawDescData = protoimpl.X.CompressGZIP(file_tensorflow_core_framework_types_proto_rawDescData)
	})
	return file_tensorflow_core_framework_types_proto_rawDescData
}

var file_tensorflow_core_framework_types_proto_enumTypes = make([]protoimpl.EnumInfo, 2)
var file_tensorflow_core_framework_types_proto_goTypes = []interface{}{
	(DataType)(0),        // 0: tensorflow.DataType
	(SpecializedType)(0), // 1: tensorflow.SpecializedType
}
var file_tensorflow_core_framework_types_proto_depIdxs = []int32{
	0, // [0:0] is the sub-list for method output_type
	0, // [0:0] is the sub-list for method input_type
	0, // [0:0] is the sub-list for extension type_name
	0, // [0:0] is the sub-list for extension extendee
	0, // [0:0] is the sub-list for field type_name
}

func init() { file_tensorflow_core_framework_types_proto_init() }
func file_tensorflow_core_framework_types_proto_init() {
	if File_tensorflow_core_framework_types_proto != nil {
		return
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_tensorflow_core_framework_types_proto_rawDesc,
			NumEnums:      2,
			NumMessages:   0,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_tensorflow_core_framework_types_proto_goTypes,
		DependencyIndexes: file_tensorflow_core_framework_types_proto_depIdxs,
		EnumInfos:         file_tensorflow_core_framework_types_proto_enumTypes,
	}.Build()
	File_tensorflow_core_framework_types_proto = out.File
	file_tensorflow_core_framework_types_proto_rawDesc = nil
	file_tensorflow_core_framework_types_proto_goTypes = nil
	file_tensorflow_core_framework_types_proto_depIdxs = nil
}

package util

import tf "github.com/codingbeard/tfkg/tensorflow/go"

func IsValidDtype(dtype tf.DataType) bool {
	valid := []tf.DataType{
		tf.Float,
		tf.Double,
		tf.Int32,
		tf.Uint32,
		tf.Uint8,
		tf.Int16,
		tf.Int8,
		tf.String,
		tf.Complex64,
		tf.Complex,
		tf.Int64,
		tf.Uint64,
		tf.Bool,
		tf.Qint8,
		tf.Quint8,
		tf.Qint32,
		tf.Bfloat16,
		tf.Qint16,
		tf.Quint16,
		tf.Uint16,
		tf.Complex128,
		tf.Half,
	}

	for _, validDtype := range valid {
		if dtype == validDtype {
			return true
		}
	}

	return false
}

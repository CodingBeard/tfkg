package preprocessor

import (
	"fmt"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"strconv"
	"strings"
)

func ReadCsvFloat32s(columns []string) interface{} {
	var floats2d [][]float32
	for _, column := range columns {
		var floats1d []float32
		for offset, b := range strings.Split(column, ",") {
			f, e := strconv.ParseFloat(b, 32)
			if e != nil {
				e := fmt.Errorf("error parsing %dth float", offset)
				panic(e)
			}
			floats1d = append(floats1d, float32(f))
		}
		floats2d = append(floats2d, floats1d)
	}

	return floats2d
}

func ReadCsvInt32s(columns []string) interface{} {
	var ints2d [][]int32
	for _, column := range columns {
		var ints1d []int32
		for offset, b := range strings.Split(column, ",") {
			i, e := strconv.Atoi(b)
			if e != nil {
				e := fmt.Errorf("error parsing %dth int", offset)
				panic(e)
			}
			ints1d = append(ints1d, int32(i))
		}
		ints2d = append(ints2d, ints1d)
	}

	return ints2d
}

func ConvertDivisorToFloat32SliceTensor(columns interface{}) (*tf.Tensor, error) {
	columnsInts, ok := columns.([][]float32)
	if !ok {
		e := fmt.Errorf("could not convert columns to [][]float32 for bools input")
		panic(e)
		return nil, e
	}
	tensor, e := tf.NewTensor(columnsInts)
	if e != nil {
		return nil, e
	}
	return tensor, nil
}

func ConvertInt32SliceToTensor(columns interface{}) (*tf.Tensor, error) {
	columnsInts, ok := columns.([][]int32)
	if !ok {
		e := fmt.Errorf("could not convert columns to [][]int32")
		return nil, e
	}
	tensor, e := tf.NewTensor(columnsInts)
	if e != nil {
		return nil, e
	}
	return tensor, nil
}

func ConvertInterfaceFloat32SliceToTensor(columns interface{}) (*tf.Tensor, error) {
	interfaceSlice, ok := columns.([]interface{})
	if !ok {
		e := fmt.Errorf("could not convert columns to []interface{}")
		return nil, e
	}
	var columnsFloats [][]float32
	for _, iSlice := range interfaceSlice {
		floatSlice, ok := iSlice.([]float32)
		if !ok {
			e := fmt.Errorf("could not convert sub slice to []float32")
			return nil, e
		}
		columnsFloats = append(columnsFloats, floatSlice)
	}
	tensor, e := tf.NewTensor(columnsFloats)
	if e != nil {
		return nil, e
	}
	return tensor, nil
}

func ReadStringNop(columns []string) interface{} {
	return columns
}

func ConvertTokenizerToInt32SliceTensor(columns interface{}) (*tf.Tensor, error) {
	columnsInts, ok := columns.([][]int32)
	if !ok {
		e := fmt.Errorf("could not convert columns to [][]int32")
		return nil, e
	}
	tensor, e := tf.NewTensor(columnsInts)
	if e != nil {
		return nil, e
	}
	return tensor, nil
}

func ConvertTokenizerToFloat32SliceTensor(columns interface{}) (*tf.Tensor, error) {
	columnsInts, ok := columns.([][]int32)
	if !ok {
		e := fmt.Errorf("could not convert columns to [][]int32")
		return nil, e
	}

	var columnsFloats [][]float32

	for _, columnInts := range columnsInts {
		var columnFloats []float32
		for _, columnInt := range columnInts {
			columnFloats = append(columnFloats, float32(columnInt))
		}
		columnsFloats = append(columnsFloats, columnFloats)
	}

	tensor, e := tf.NewTensor(columnsFloats)
	if e != nil {
		return nil, e
	}
	return tensor, nil
}

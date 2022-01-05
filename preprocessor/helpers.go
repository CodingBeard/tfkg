package preprocessor

import (
	"fmt"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"image"
	"image/color"
	_ "image/jpeg"
	"os"
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

func ReadJpg(columns []string) interface{} {
	var images []image.Image
	for _, column := range columns {
		f, e := os.Open(column)
		if e != nil {
			panic(e)
		}
		img, _, e := image.Decode(f)
		if e != nil {
			panic(e)
		}
		images = append(images, img)
	}

	return images
}

func ConvertImageToFloat32SliceTensor(columns interface{}) (*tf.Tensor, error) {
	images, ok := columns.([]ProcessedImage)
	if !ok {
		e := fmt.Errorf("could not convert columns to [][]float32 for bools input")
		panic(e)
		return nil, e
	}
	var imagesFloats [][][][]float32
	for _, img := range images {
		height := img.Image.Bounds().Dy()
		width := img.Image.Bounds().Dx()
		var imageFloats [][][]float32
		for y := 0; y < height; y++ {
			rowFloats := make([][]float32, width)
			for x := 0; x < width; x++ {
				switch offset := img.Image.At(x, y).(type) {
				case color.Gray:
					rowFloats[x] = append(rowFloats[x], float32(offset.Y)/255)
				case color.RGBA:
					rowFloats[x] = append(rowFloats[x], float32(offset.R)/255)
					rowFloats[x] = append(rowFloats[x], float32(offset.G)/255)
					rowFloats[x] = append(rowFloats[x], float32(offset.B)/255)
					if img.Color == ImageColorRGBA {
						rowFloats[x] = append(rowFloats[x], float32(offset.A)/255)
					}
				default:
					panic(fmt.Sprintf("Unsupported color type: %#v", offset))
				}
			}
			imageFloats = append(imageFloats, rowFloats)
		}
		imagesFloats = append(imagesFloats, imageFloats)
	}
	tensor, e := tf.NewTensor(imagesFloats)
	if e != nil {
		return nil, e
	}
	return tensor, nil
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

func ConvertInterfaceToInt32SliceTensor(columns interface{}) (*tf.Tensor, error) {
	columnsInterfaces, ok := columns.([]interface{})
	if !ok {
		e := fmt.Errorf("could not convert columns to []int32")
		return nil, e
	}
	var columnInts [][]int32
	for _, columnInterface := range columnsInterfaces {
		columnInts = append(columnInts, []int32{columnInterface.(int32)})
	}
	tensor, e := tf.NewTensor(columnInts)
	if e != nil {
		return nil, e
	}
	return tensor, nil
}

func ConvertInterfaceToFloat32SliceTensor(columns interface{}) (*tf.Tensor, error) {
	columnsInterfaces, ok := columns.([]interface{})
	if !ok {
		e := fmt.Errorf("could not convert columns to []int32")
		return nil, e
	}
	var columnInts [][]float32
	for _, columnInterface := range columnsInterfaces {
		columnInts = append(columnInts, []float32{columnInterface.(float32)})
	}
	tensor, e := tf.NewTensor(columnInts)
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

func ConvertInterfaceInt32SliceToTensor(columns interface{}) (*tf.Tensor, error) {
	interfaceSlice, ok := columns.([]interface{})
	if !ok {
		e := fmt.Errorf("could not convert columns to []interface{}")
		return nil, e
	}
	var columnsFloats [][]int32
	for _, iSlice := range interfaceSlice {
		floatSlice, ok := iSlice.([]int32)
		if !ok {
			e := fmt.Errorf("could not convert sub slice to []int32")
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

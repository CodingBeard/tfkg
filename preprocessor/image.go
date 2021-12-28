package preprocessor

import (
	"fmt"
	"github.com/codingbeard/cberrors"
	"github.com/nfnt/resize"
	"image"
	"image/draw"
)

type ImageColor string

var (
	ImageColorGray ImageColor = "gray"
	ImageColorRGBA ImageColor = "rgba"
	ImageColorRGB  ImageColor = "rgb"
)

type Image struct {
	colorMode     ImageColor
	resizeX       int
	resizeY       int
	interpolation resize.InterpolationFunction

	errorHandler *cberrors.ErrorsContainer
}

type ImageConfig struct {
	ColorMode     ImageColor
	ResizeX       int
	ResizeY       int
	Interpolation resize.InterpolationFunction
}

func NewImage(
	errorHandler *cberrors.ErrorsContainer,
	configs ...ImageConfig,
) *Image {
	config := ImageConfig{}
	if len(configs) > 0 {
		config = configs[0]
	}
	if config.ColorMode == "" {
		config.ColorMode = ImageColorRGB
	}
	return &Image{
		colorMode:     config.ColorMode,
		resizeX:       config.ResizeX,
		resizeY:       config.ResizeY,
		interpolation: config.Interpolation,
		errorHandler:  errorHandler,
	}
}

type ProcessedImage struct {
	Color ImageColor
	Image image.Image
}

func (i *Image) Process(img image.Image) (ProcessedImage, error) {
	if i.resizeX != 0 && i.resizeY != 0 {
		img = resize.Resize(uint(i.resizeX), uint(i.resizeY), img, i.interpolation)
	}
	if i.colorMode == ImageColorGray {
		processedImg := image.NewGray(img.Bounds())
		for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
			for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
				processedImg.Set(x, y, img.At(x, y))
			}
		}
		return ProcessedImage{
			Color: i.colorMode,
			Image: processedImg,
		}, nil
	} else if i.colorMode == ImageColorRGB || i.colorMode == ImageColorRGBA {
		b := img.Bounds()
		processedImg := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
		draw.Draw(processedImg, processedImg.Bounds(), img, b.Min, draw.Src)
		return ProcessedImage{
			Color: i.colorMode,
			Image: processedImg,
		}, nil
	}
	e := fmt.Errorf("unknown image color mode")
	i.errorHandler.Error(e)
	return ProcessedImage{}, e
}

package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LResizing struct {
	cropToAspectRatio bool
	dtype             DataType
	height            float64
	inputs            []Layer
	interpolation     string
	name              string
	shape             tf.Shape
	trainable         bool
	width             float64
}

func Resizing(height float64, width float64) *LResizing {
	return &LResizing{
		cropToAspectRatio: false,
		dtype:             Float32,
		height:            height,
		interpolation:     "bilinear",
		name:              UniqueName("resizing"),
		trainable:         true,
		width:             width,
	}
}

func (l *LResizing) SetCropToAspectRatio(cropToAspectRatio bool) *LResizing {
	l.cropToAspectRatio = cropToAspectRatio
	return l
}

func (l *LResizing) SetDtype(dtype DataType) *LResizing {
	l.dtype = dtype
	return l
}

func (l *LResizing) SetInterpolation(interpolation string) *LResizing {
	l.interpolation = interpolation
	return l
}

func (l *LResizing) SetName(name string) *LResizing {
	l.name = name
	return l
}

func (l *LResizing) SetShape(shape tf.Shape) *LResizing {
	l.shape = shape
	return l
}

func (l *LResizing) SetTrainable(trainable bool) *LResizing {
	l.trainable = trainable
	return l
}

func (l *LResizing) GetShape() tf.Shape {
	return l.shape
}

func (l *LResizing) GetDtype() DataType {
	return l.dtype
}

func (l *LResizing) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LResizing) GetInputs() []Layer {
	return l.inputs
}

func (l *LResizing) GetName() string {
	return l.name
}

type jsonConfigLResizing struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LResizing) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range l.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	return jsonConfigLResizing{
		ClassName: "Resizing",
		Name:      l.name,
		Config: map[string]interface{}{
			"crop_to_aspect_ratio": l.cropToAspectRatio,
			"dtype":                l.dtype.String(),
			"height":               l.height,
			"interpolation":        l.interpolation,
			"name":                 l.name,
			"trainable":            l.trainable,
			"width":                l.width,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LResizing) GetCustomLayerDefinition() string {
	return ``
}

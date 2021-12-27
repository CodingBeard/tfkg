package layer

import tf "github.com/galeone/tensorflow/tensorflow/go"

type LMaxPooling3D struct {
	dataFormat interface{}
	dtype      DataType
	inputs     []Layer
	name       string
	padding    string
	poolSize   []interface{}
	shape      tf.Shape
	strides    interface{}
	trainable  bool
}

func MaxPooling3D() *LMaxPooling3D {
	return &LMaxPooling3D{
		dataFormat: nil,
		dtype:      Float32,
		name:       UniqueName("max_pooling3d"),
		padding:    "valid",
		poolSize:   []interface{}{2, 2, 2},
		strides:    nil,
		trainable:  true,
	}
}

func (l *LMaxPooling3D) SetDataFormat(dataFormat interface{}) *LMaxPooling3D {
	l.dataFormat = dataFormat
	return l
}

func (l *LMaxPooling3D) SetDtype(dtype DataType) *LMaxPooling3D {
	l.dtype = dtype
	return l
}

func (l *LMaxPooling3D) SetName(name string) *LMaxPooling3D {
	l.name = name
	return l
}

func (l *LMaxPooling3D) SetPadding(padding string) *LMaxPooling3D {
	l.padding = padding
	return l
}

func (l *LMaxPooling3D) SetPoolSize(poolSize []interface{}) *LMaxPooling3D {
	l.poolSize = poolSize
	return l
}

func (l *LMaxPooling3D) SetShape(shape tf.Shape) *LMaxPooling3D {
	l.shape = shape
	return l
}

func (l *LMaxPooling3D) SetStrides(strides interface{}) *LMaxPooling3D {
	l.strides = strides
	return l
}

func (l *LMaxPooling3D) SetTrainable(trainable bool) *LMaxPooling3D {
	l.trainable = trainable
	return l
}

func (l *LMaxPooling3D) GetShape() tf.Shape {
	return l.shape
}

func (l *LMaxPooling3D) GetDtype() DataType {
	return l.dtype
}

func (l *LMaxPooling3D) SetInputs(inputs ...Layer) Layer {
	l.inputs = inputs
	return l
}

func (l *LMaxPooling3D) GetInputs() []Layer {
	return l.inputs
}

func (l *LMaxPooling3D) GetName() string {
	return l.name
}

type jsonConfigLMaxPooling3D struct {
	ClassName    string                 `json:"class_name"`
	Name         string                 `json:"name"`
	Config       map[string]interface{} `json:"config"`
	InboundNodes [][][]interface{}      `json:"inbound_nodes"`
}

func (l *LMaxPooling3D) GetKerasLayerConfig() interface{} {
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
	return jsonConfigLMaxPooling3D{
		ClassName: "MaxPooling3D",
		Name:      l.name,
		Config: map[string]interface{}{
			"data_format": l.dataFormat,
			"dtype":       l.dtype.String(),
			"name":        l.name,
			"padding":     l.padding,
			"pool_size":   l.poolSize,
			"strides":     l.strides,
			"trainable":   l.trainable,
		},
		InboundNodes: inboundNodes,
	}
}

func (l *LMaxPooling3D) GetCustomLayerDefinition() string {
	return ``
}

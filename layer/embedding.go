package layer

import (
	tf "github.com/galeone/tensorflow/tensorflow/go"
)

type Embedding struct {
	numUniqueWords int
	embeddingDim   int
	shape          tf.Shape
	name           string
	dtype          DataType
	inputs         []Layer
}

type EmbeddingConfig struct {
	Name string
}

func NewEmbedding(numUniqueWords int, embeddingDim int, optionalConfig ...EmbeddingConfig) func(inputs ...Layer) Layer {
	var config EmbeddingConfig
	if len(optionalConfig) == 1 {
		config = optionalConfig[0]
	}

	if config.Name == "" {
		config.Name = uniqueName("embedding")
	}

	return func(inputs ...Layer) Layer {
		return &Embedding{
			numUniqueWords: numUniqueWords,
			embeddingDim:   embeddingDim,
			name:           config.Name,
			inputs:         inputs,
			dtype:          Float32,
		}
	}
}

func (em *Embedding) GetShape() tf.Shape {
	return em.shape
}

func (em *Embedding) GetDtype() DataType {
	return em.dtype
}

func (em *Embedding) SetInput(inputs []Layer) {
	em.dtype = Float32
	em.inputs = inputs
}

func (em *Embedding) GetInputs() []Layer {
	return em.inputs
}

func (em *Embedding) GetName() string {
	return em.name
}

type kerasEmbeddingConfig struct {
	ClassName string `json:"class_name"`
	Config    struct {
		Name                  string        `json:"name"`
		Trainable             bool          `json:"trainable"`
		BatchInputShape       []interface{} `json:"batch_input_shape"`
		Dtype                 string        `json:"dtype"`
		InputDim              int           `json:"input_dim"`
		OutputDim             int           `json:"output_dim"`
		EmbeddingsInitializer struct {
			ClassName string `json:"class_name"`
			Config    struct {
				Minval float64     `json:"minval"`
				Maxval float64     `json:"maxval"`
				Seed   interface{} `json:"seed"`
			} `json:"config"`
		} `json:"embeddings_initializer"`
		EmbeddingsRegularizer interface{} `json:"embeddings_regularizer"`
		ActivityRegularizer   interface{} `json:"activity_regularizer"`
		EmbeddingsConstraint  interface{} `json:"embeddings_constraint"`
		MaskZero              bool        `json:"mask_zero"`
		InputLength           interface{} `json:"input_length"`
	} `json:"config"`
	Name         string            `json:"name"`
	InboundNodes [][][]interface{} `json:"inbound_nodes"`
}

func (em *Embedding) GetKerasLayerConfig() interface{} {
	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range em.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}
	config := kerasEmbeddingConfig{
		ClassName: "Embedding",
		Config: struct {
			Name                  string        `json:"name"`
			Trainable             bool          `json:"trainable"`
			BatchInputShape       []interface{} `json:"batch_input_shape"`
			Dtype                 string        `json:"dtype"`
			InputDim              int           `json:"input_dim"`
			OutputDim             int           `json:"output_dim"`
			EmbeddingsInitializer struct {
				ClassName string `json:"class_name"`
				Config    struct {
					Minval float64     `json:"minval"`
					Maxval float64     `json:"maxval"`
					Seed   interface{} `json:"seed"`
				} `json:"config"`
			} `json:"embeddings_initializer"`
			EmbeddingsRegularizer interface{} `json:"embeddings_regularizer"`
			ActivityRegularizer   interface{} `json:"activity_regularizer"`
			EmbeddingsConstraint  interface{} `json:"embeddings_constraint"`
			MaskZero              bool        `json:"mask_zero"`
			InputLength           interface{} `json:"input_length"`
		}{
			Name:            em.name,
			Trainable:       true,
			BatchInputShape: []interface{}{nil, nil},
			Dtype:           string(em.dtype),
			InputDim:        em.numUniqueWords,
			OutputDim:       em.embeddingDim,
			EmbeddingsInitializer: struct {
				ClassName string `json:"class_name"`
				Config    struct {
					Minval float64     `json:"minval"`
					Maxval float64     `json:"maxval"`
					Seed   interface{} `json:"seed"`
				} `json:"config"`
			}{
				ClassName: "RandomUniform",
				Config: struct {
					Minval float64     `json:"minval"`
					Maxval float64     `json:"maxval"`
					Seed   interface{} `json:"seed"`
				}{
					Minval: -0.05,
					Maxval: 0.05,
					Seed:   nil,
				},
			},
			EmbeddingsRegularizer: nil,
			ActivityRegularizer:   nil,
			EmbeddingsConstraint:  nil,
			MaskZero:              false,
			InputLength:           em.inputs[0].GetShape().Size(1),
		},
		Name:         em.name,
		InboundNodes: inboundNodes,
	}

	return config
}

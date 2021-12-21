package initializer

type Initializer interface {
	GetKerasLayerConfig() interface{}
}

type NilInitializer struct{}

func (n *NilInitializer) GetKerasLayerConfig() interface{} {
	return nil
}

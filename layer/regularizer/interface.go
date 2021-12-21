package regularizer

type Regularizer interface {
	GetKerasLayerConfig() interface{}
}

type NilRegularizer struct{}

func (n *NilRegularizer) GetKerasLayerConfig() interface{} {
	return nil
}

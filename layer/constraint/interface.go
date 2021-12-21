package constraint

type Constraint interface {
	GetKerasLayerConfig() interface{}
}

type NilConstraint struct{}

func (n *NilConstraint) GetKerasLayerConfig() interface{} {
	return nil
}

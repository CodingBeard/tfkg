module github.com/codingbeard/tfkg

go 1.17

replace github.com/codingbeard/tfkg/tensorflow/go v0.0.0-20210519172502-4018d721b591 => ./tensorflow/go

require (
	github.com/codingbeard/tfkg/tensorflow/go v0.0.0-20210519172502-4018d721b591
)

require (
	github.com/golang/protobuf v1.5.2 // indirect
	google.golang.org/protobuf v1.26.0 // indirect
)

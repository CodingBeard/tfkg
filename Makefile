SHELL:=/bin/bash
PROJECT_NAME=tfkg
GO_FILES=./...

.SILENT: examples-iris
.PHONY: examples-iris

init-docker-m1:
	docker pull tensorflow/tensorflow:devel-gpu@sha256:1452331ddc5c1995b508114ec9bae0812e17ce14342bd67e08244a07c0d5a5cb
	docker buildx build docker/tf-jupyter-golang-m1 --platform=linux/amd64 -t tf-jupyter-golang

init-docker:
	docker pull tensorflow/tensorflow:2.6.0-gpu-jupyter
	docker build docker/tf-jupyter-golang -t tf-jupyter-golang

examples-iris:
	go generate ./...
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg && go run examples/iris/main.go"

examples-iris-gpu:
	go generate ./...
	docker-compose up -d tf-jupyter-golang-gpu
	docker-compose exec tf-jupyter-golang-gpu sh -c "cd /go/src/tfkg && go run examples/iris/main.go"

examples-multiple-inputs:
	go generate ./...
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg && go run examples/multiple_inputs/main.go"

examples-multiple-inputs-gpu:
	go generate ./...
	docker-compose up -d tf-jupyter-golang-gpu
	docker-compose exec tf-jupyter-golang-gpu sh -c "cd /go/src/tfkg && go run examples/multiple_inputs/main.go"

examples-jobs:
	go generate ./...
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg && go run examples/jobs/main.go"

examples-jobs-gpu:
	go generate ./...
	docker-compose up -d tf-jupyter-golang-gpu
	docker-compose exec tf-jupyter-golang-gpu sh -c "cd /go/src/tfkg && go run examples/jobs/main.go"

examples-class-weights:
	go generate ./...
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg && go run examples/class_weights/main.go"

test-python:
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg && python test.py"
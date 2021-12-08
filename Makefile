SHELL:=/bin/bash
PROJECT_NAME=tfkg
GO_FILES=./...

.SILENT: examples-iris
.PHONY: examples-iris

init-docker-m1:
	docker buildx build docker/tf-jupyter-golang-m1 --platform=linux/amd64 -t tf-jupyter-golang

init-docker:
	docker build docker/tf-jupyter-golang -t tf-jupyter-golang

examples-iris:
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg && go run examples/iris/main.go"

test-python:
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /tfkg && cat config.txt | python test.py"
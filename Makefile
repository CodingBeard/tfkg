SHELL:=/bin/bash
PROJECT_NAME=tfkg
GO_FILES=./...

.SILENT: web
.PHONY: web

init-docker-m1:
	docker pull tensorflow/tensorflow:devel-gpu@sha256:1452331ddc5c1995b508114ec9bae0812e17ce14342bd67e08244a07c0d5a5cb
	docker buildx build docker/tf-jupyter-golang-m1 --platform=linux/amd64 -t tf-jupyter-golang

init-docker:
	docker pull tensorflow/tensorflow:2.6.0-gpu-jupyter
	docker build docker/tf-jupyter-golang -t tf-jupyter-golang

RSYNC_REMOTE?=""
RSYNC_REMOTE_PATH?=""
REMOTE?=""
COMMAND?=""
DELETE?=""
remote:
	rsync $(DELETE) --recursive --relative --exclude=logs --exclude=node_modules --exclude=tensorflow --exclude=examples/*/training-cache --exclude=examples/*/data * $(RSYNC_REMOTE):$(RSYNC_REMOTE_PATH) || true
	ssh -t $(REMOTE) "$(COMMAND)"

examples-iris:
	go generate ./...
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg/examples/iris && go run main.go"

examples-iris-raw:
	go generate ./...
	cd examples/iris && go run main.go

examples-iris-gpu:
	go generate ./...
	docker-compose up -d tf-jupyter-golang-gpu
	docker-compose exec tf-jupyter-golang-gpu sh -c "cd /go/src/tfkg/examples/iris && go run main.go"

examples-multiple-inputs:
	go generate ./...
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg/examples/multiple_inputs && go run main.go"

examples-multiple-inputs-gpu:
	go generate ./...
	docker-compose up -d tf-jupyter-golang-gpu
	docker-compose exec tf-jupyter-golang-gpu sh -c "cd /go/src/tfkg/examples/multiple_inputs && go run main.go"

examples-multiple-inputs-raw:
	go generate ./...
	cd examples/multiple_inputs && go run main.go

examples-jobs:
	go generate ./...
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg/examples/jobs && go run main.go"

examples-jobs-gpu:
	go generate ./...
	docker-compose up -d tf-jupyter-golang-gpu
	docker-compose exec tf-jupyter-golang-gpu sh -c "cd /go/src/tfkg/examples/jobs && go run main.go"

examples-jobs-raw:
	go generate ./...
	cd examples/jobs && go run main.go

examples-class-weights:
	go generate ./...
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg/examples/class_weights && go run main.go"

examples-class-weights-raw:
	go generate ./...
	cd examples/class_weights && go run main.go

examples-sign:
	go generate ./...
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg/examples/sign && go run main.go"

examples-sign-gpu:
	go generate ./...
	docker-compose up -d tf-jupyter-golang-gpu
	docker-compose exec tf-jupyter-golang-gpu sh -c "cd /go/src/tfkg/examples/sign && go run main.go"

examples-sign-raw:
	go generate ./...
	cd examples/sign && go run main.go

examples-transfer:
	go generate ./...
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg/examples/transfer_learning && go run main.go"

examples-transfer-gpu:
	go generate ./...
	docker-compose up -d tf-jupyter-golang-gpu
	docker-compose exec tf-jupyter-golang-gpu sh -c "cd /go/src/tfkg/examples/transfer_learning && go run main.go"

examples-transfer-raw:
	go generate ./...
	cd examples/transfer_learning && go run main.go

examples-vanilla:
	go generate ./...
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg/examples/vanilla && python generate_vanilla_model.py && go run main.go"

examples-vanilla-gpu:
	go generate ./...
	docker-compose up -d tf-jupyter-golang-gpu
	docker-compose exec tf-jupyter-golang-gpu sh -c "cd /go/src/tfkg/examples/vanilla && python generate_vanilla_model.py && go run main.go"

examples-vanilla-raw:
	go generate ./...
	cd examples/vanilla && python generate_vanilla_model.py && go run main.go

examples-bench:
	go generate ./...
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg/examples/bench && go run main.go && python bench_python.py"

examples-bench-gpu:
	go generate ./...
	docker-compose up -d tf-jupyter-golang-gpu
	docker-compose exec tf-jupyter-golang-gpu sh -c "cd /go/src/tfkg/examples/bench && go run main.go && python bench_python.py"

examples-bench-raw:
	go generate ./...
	cd examples/bench && go run main.go && python bench_python.py

test-python:
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg && python test.py"

dev-generate-keras-objects-python:
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg && python generate/python/generate_keras_objects.py"

dev-generate-keras-objects-golang:
	docker-compose up -d tf-jupyter-golang
	docker-compose exec tf-jupyter-golang sh -c "cd /go/src/tfkg/generate/python/ && go run generate.go"

web: frontend-build
	docker-compose up -d web
	echo Started web service on port 8082 reading model subdirs in dir: ./logs

dev-web-gin:
	docker-compose up -d web-development
	echo Starting web-development on port 8082
	docker-compose exec web-development sh -c "pkill -f gin" || true
	docker-compose exec web-development sh -c "cd /go/src/tfkg/web && rm -f tfkg && MODE=web gin --immediate --bin tfkg --port 80 --appPort 3010 run main.go"

dev-web-refresh:
	$(MAKE) -j2 dev-web-gin frontend-reload

#build the frontend
frontend-build:
	echo Building frontend
	$(MAKE) -j1 npm-update
	echo installing gulp if needed, and running gulp
	docker-compose exec frontend sh -c "pkill -f gulp" || true
	docker-compose exec frontend sh -c "cd /app && npm update && if which gulp; then echo Found gulp; else npm install -g gulp-cli; fi && gulp --env=prod"

npm-update:
	echo Starting docker frontend build container
	docker-compose up -d frontend
	echo Running npm update
	docker-compose exec frontend sh -c "cd /app && npm update"

#build the frontend and watch for changes, rebuilding when detected
frontend-watch:
	echo Starting docker frontend build container
	docker-compose up -d frontend
	echo installing gulp if needed, and running gulp
	docker-compose exec frontend sh -c "pkill -f gulp" || true
	docker-compose exec frontend sh -c "cd /app && if which gulp; then echo Found gulp; else npm install -g gulp-cli; fi && gulp watch --env=dev"

#build the frontend and watch for changes, rebuilding code and reloading the browser when detected
frontend-reload:
	echo Starting docker frontend build container
	docker-compose up -d frontend
	echo installing gulp if needed, and running gulp
	docker-compose exec frontend sh -c "pkill -f gulp" || true
	docker-compose exec frontend sh -c "cd /app && if which gulp; then echo Found gulp; else npm --no-color install -g gulp-cli; fi && gulp reload --env=dev"

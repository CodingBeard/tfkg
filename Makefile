SHELL:=/bin/bash
PROJECT_NAME=tfkg
GO_FILES=./...

.SILENT: examples-iris
.PHONY: examples-iris

examples-iris:
	docker-compose exec -T tf-jupyter-golang sh -c "cd /go/src/tfkg && go run examples/iris/main.go"

test-python:
	docker-compose exec -T tf-jupyter-golang sh -c "cd /tfkg && python test.py"
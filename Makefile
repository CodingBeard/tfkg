SHELL:=/bin/bash
PROJECT_NAME=tfkg
GO_FILES=./...

.SILENT: test create-model
.PHONY: test create-model

test:
	docker-compose exec -T golang sh -c "pkill -f \"go run\"" || true
	docker-compose exec -T golang sh -c "cd /go/src/tfkg && go run ./examples/test.go"

create-model:
	docker-compose exec -T tf-jupyter sh -c "cd /tfkg && python create_base_model.py"

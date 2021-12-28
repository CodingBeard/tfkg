module github.com/codingbeard/tfkg

go 1.17

replace github.com/galeone/tensorflow/tensorflow/go v0.0.0-20210519172502-4018d721b591 => github.com/codingbeard/tensorflow/tensorflow/go v0.0.0-20210519172502-4018d721b591

//replace github.com/codingbeard/cbweb v0.10.9 => ./.codingbeard/cbweb

require (
	github.com/GeertJohan/go.rice v1.0.2
	github.com/codingbeard/cberrors v0.0.3
	github.com/codingbeard/cblog v0.0.5
	github.com/codingbeard/cbutil v0.2.2
	github.com/codingbeard/cbweb v0.10.9
	github.com/fasthttp/router v1.4.4
	github.com/galeone/tensorflow/tensorflow/go v0.0.0-20210519172502-4018d721b591
	github.com/karrick/godirwalk v1.16.1
	github.com/remeh/sizedwaitgroup v1.0.0
	github.com/valyala/fasthttp v1.31.0
)

require (
	github.com/andybalholm/brotli v1.0.2 // indirect
	github.com/codingbeard/go-logger v0.0.0-20201005090617-a00c36603e2d // indirect
	github.com/daaku/go.zipexe v1.0.0 // indirect
	github.com/didip/tollbooth v1.0.0 // indirect
	github.com/didip/tollbooth_fasthttp v0.0.0-20170910065828-cfa276ddefe2 // indirect
	github.com/fatih/camelcase v1.0.0 // indirect
	github.com/golang/protobuf v1.5.2 // indirect
	github.com/klauspost/compress v1.13.4 // indirect
	github.com/nfnt/resize v0.0.0-20180221191011-83c6a9932646 // indirect
	github.com/patrickmn/go-cache v2.1.0+incompatible // indirect
	github.com/savsgio/gotils v0.0.0-20210921075833-21a6215cb0e4 // indirect
	github.com/twpayne/go-jsonstruct v0.0.0-20200905114252-a1027bf3a425 // indirect
	github.com/valyala/bytebufferpool v1.0.0 // indirect
	golang.org/x/time v0.0.0-20191024005414-555d28b269f0 // indirect
	google.golang.org/protobuf v1.26.0 // indirect
	gopkg.in/yaml.v3 v3.0.0-20210107192922-496545a6307b // indirect
)

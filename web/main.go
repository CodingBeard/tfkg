package main

import (
	rice "github.com/GeertJohan/go.rice"
	"github.com/codingbeard/cberrors"
	"github.com/codingbeard/cberrors/iowriterprovider"
	"github.com/codingbeard/cblog"
	"github.com/codingbeard/cbweb"
	"github.com/codingbeard/cbweb/module/cbwebcommon"
	"github.com/codingbeard/tfkg/web/module/home"
	"github.com/codingbeard/tfkg/web/module/metrics"
	"github.com/valyala/fasthttp"
	"html/template"
	"os"
)

func main() {
	port := os.Getenv("PORT")
	if len(port) > 0 {
		port = ":" + port
	} else {
		port = ":80"
	}

	// Create a logger pointed at the save dir
	logger, e := cblog.NewLogger(cblog.LoggerConfig{
		LogLevel:           cblog.DebugLevel,
		Format:             "%{time:2006-01-02 15:04:05.000} : %{file}:%{line} : %{message}",
		LogToFile:          true,
		FilePath:           "web.log",
		FilePerm:           os.ModePerm,
		LogToStdOut:        true,
		SetAsDefaultLogger: true,
	})
	if e != nil {
		panic(e)
	}

	// Error handler with stack traces
	errorHandler := cberrors.NewErrorContainer(iowriterprovider.New(logger))

	methods := []rice.LocateMethod{rice.LocateEmbedded, rice.LocateAppended, rice.LocateFS}
	config := rice.Config{LocateOrder: methods}

	riceWebAssets, e := config.FindBox("public")
	if e != nil {
		panic(e)
	}
	riceTemplates, e := config.FindBox("module")
	if e != nil {
		panic(e)
	}

	cbWebCommonModule := &cbwebcommon.Module{
		Env:          "dev",
		BrandName:    "TFKG",
		WebAssets:    riceWebAssets.HTTPBox(),
		TemplatesBox: riceTemplates.HTTPBox(),
		ErrorHandler: errorHandler,
		Logger:       logger,
	}

	cbWebCommonModule.SetDefaults()

	navItems := func(ctx *fasthttp.RequestCtx) []cbweb.NavItem {
		navItems := cbweb.NavItemCollection{
			{Title: "Home", SubNavItems: []cbweb.NavItem{
				{
					Permitted: true,
					Title:     "Home",
					Src:       template.URL(home.HomeRoute),
				},
			}},
			{Title: "Metrics", SubNavItems: []cbweb.NavItem{
				{
					Permitted: true,
					Title:     "Models",
					Src:       template.URL(metrics.ModelsRoute),
				},
			}},
		}

		return navItems.FilterPermitted()
	}

	modules := []cbweb.Module{
		cbWebCommonModule,
		&home.Module{
			Common:       cbWebCommonModule,
			ErrorHandler: errorHandler,
			Logger:       logger,
			NavItems:     navItems,
		},
		&metrics.Module{
			Common:       cbWebCommonModule,
			ErrorHandler: errorHandler,
			Logger:       logger,
			NavItems:     navItems,
		},
	}

	logger.InfoF("root", "Initialising http server")

	server := cbweb.NewServer(
		cbweb.Dependencies{
			Port:         port,
			ErrorHandler: errorHandler,
		},
		modules...,
	)

	e = server.Start()

	if e != nil {
		errorHandler.Error(e)
	}
}

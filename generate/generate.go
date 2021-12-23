package main

import (
	"fmt"
	"io/ioutil"
	"os"
)

func main() {
	contents, e := ioutil.ReadFile("../model/model.py")
	if e != nil {
		panic(e)
	}
	e = ioutil.WriteFile("../model/python_generated.go", []byte(fmt.Sprintf(`package model

import (
	"strings"
)

// This code is generated automatically using "go generate ./..." from model/model.py. DO NOT EDIT manually.
func GetPythonCode(customDefinitions []string) string {
	return strings.ReplaceAll(%s%s%s, "# tfkg-custom-definitions", strings.Join(customDefinitions, "\n"))
}
`, "`", string(contents), "`")), os.ModePerm)
	if e != nil {
		panic(e)
	}
}

package main

import (
	"fmt"
	"io/ioutil"
	"os"
)

func main() {
	tfkgContent, e := ioutil.ReadFile("../model/tfkg_model.py")
	if e != nil {
		panic(e)
	}
	vanillaContent, e := ioutil.ReadFile("../model/vanilla_model.py")
	if e != nil {
		panic(e)
	}
	e = ioutil.WriteFile("../model/python_generated.go", []byte(fmt.Sprintf(`package model

import (
	"strings"
)

// This code is generated automatically using "go generate ./..." from model/tfkg_model.py. DO NOT EDIT manually.
func GetTfkgPythonCode(customDefinitions []string) string {
	return strings.ReplaceAll(%s%s%s, "# tfkg-custom-definitions", strings.Join(customDefinitions, "\n"))
}

func GetVanillaPythonCode() string {
	return %s%s%s
}
`, "`", string(tfkgContent), "`", "`", string(vanillaContent), "`")), os.ModePerm)
	if e != nil {
		panic(e)
	}
}

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
	e = ioutil.WriteFile("../model/python_generated.go", []byte(fmt.Sprintf(`
package model

// This code is generated automaticall using go generate. DO NOT EDIT manually
func GetPythonCode() string {
	return %s%s%s
}
`, "`", string(contents), "`")), os.ModePerm)
	if e != nil {
		panic(e)
	}
}

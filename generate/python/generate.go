package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

// Only for use in tfkg development E.G. adding support for new keras versions.
// This generates all the golang code for keras layer config generation
// Run generate_keras_objects.py first
// This is very, very dirty. But "it works"

type objectJson struct {
	Type           string                 `json:"type"`
	Name           string                 `json:"name"`
	RequiredParams []interface{}          `json:"required"`
	OptionalParams []interface{}          `json:"optional"`
	Config         map[string]interface{} `json:"config"`
}

func main() {
	configFileBytes, e := ioutil.ReadFile("objects.json")
	if e != nil {
		panic(e)
	}
	var objects []objectJson
	e = json.Unmarshal(configFileBytes, &objects)
	if e != nil {
		panic(e)
	}
	for _, object := range objects {
		fmt.Println(object.Type, object.Name)
		if object.Type == "optimizer" {
			f := newFileGenerator(
				object,
				"../../optimizer",
				&parameter{
					Name: "name",
				},
			)
			f.generate()
		} else if object.Type == "initializer" {
			f := newFileGenerator(
				object,
				"../../layer/initializer",
				&parameter{
					Name: "name",
				},
			)
			f.generate()
		} else if object.Type == "regularizer" {
			f := newFileGenerator(
				object,
				"../../layer/regularizer",
				&parameter{
					Name: "name",
				},
			)
			f.generate()
		} else if object.Type == "constraint" {
			f := newFileGenerator(
				object,
				"../../layer/constraint",
				&parameter{
					Name: "name",
				},
			)
			f.generate()
		} else if object.Type == "layer" {
			f := newFileGenerator(
				object,
				"../../layer",
				&parameter{
					Name:       "shape",
					StringType: "tf.Shape",
				},
				&parameter{
					Name:       "inputs",
					StringType: "[]Layer",
				},
				&parameter{
					Name: "name",
				},
			)
			f.generate()
		}
	}

	_, e = exec.Command("go", "fmt", "github.com/codingbeard/tfkg/optimizer").Output()
	if e != nil {
		panic(e)
	}
	_, e = exec.Command("go", "fmt", "github.com/codingbeard/tfkg/layer/initializer").Output()
	if e != nil {
		panic(e)
	}
	_, e = exec.Command("go", "fmt", "github.com/codingbeard/tfkg/layer/regularizer").Output()
	if e != nil {
		panic(e)
	}
	_, e = exec.Command("go", "fmt", "github.com/codingbeard/tfkg/layer/constraint").Output()
	if e != nil {
		panic(e)
	}
	_, e = exec.Command("go", "fmt", "github.com/codingbeard/tfkg/layer").Output()
	if e != nil {
		panic(e)
	}
}

type parameter struct {
	ObjectName   string
	Name         string
	Default      interface{}
	StringType   string
	IsRequired   bool
	IsAdditional bool
}

func (p *parameter) getCamelCaseName() string {
	str := p.Name
	str = strings.ReplaceAll(str, "_", " ")
	str = strings.Title(str)
	str = strings.ReplaceAll(str, " ", "")
	return strings.ToLower(string(str[0])) + str[1:]
}

func (p *parameter) getGolangType() string {
	stringValue := ""
	if strings.HasSuffix(p.Name, "constraint") {
		stringValue = "constraint.Constraint"
	} else if strings.HasSuffix(p.Name, "initializer") && p.ObjectName != "LRandomFourierFeatures" {
		stringValue = "initializer.Initializer"
	} else if strings.HasSuffix(p.Name, "regularizer") {
		stringValue = "regularizer.Regularizer"
	} else if strings.HasSuffix(p.Name, "activation") {
		stringValue = "string"
	} else {
		if p.StringType != "" {
			stringValue = p.StringType
		} else {
			stringValue = fmt.Sprintf("%T", p.Default)
		}
		if stringValue == "<nil>" {
			stringValue = "interface{}"
		}
		if p.Name == "dtype" {
			stringValue = "DataType"
		}
		if p.Name == "name" {
			stringValue = "string"
		}
	}
	return stringValue
}

func (p *parameter) getStringDefaultValue() string {
	stringValue := fmt.Sprintf("%#v", p.Default)
	if stringValue == "<nil>" {
		stringValue = "nil"
	}
	if p.Name == "activation" && stringValue == "nil" {
		stringValue = `"linear"`
	} else if p.Name == "embeddings_initializer" && fmt.Sprint(p.Default) == "uniform" {
		stringValue = "initializer.RandomUniform()"
	} else if strings.HasSuffix(p.Name, "constraint") {
		if stringValue != "nil" {
			if config, ok := p.Default.(map[string]interface{}); ok {
				stringValue = fmt.Sprintf("constraint.%s()", config["class_name"])
			} else {
				stringValue = fmt.Sprintf("constraint.%s()", strings.Title(snakeCaseToCamelCase(fmt.Sprint(p.Default))))
			}
		} else {
			stringValue = "&constraint.NilConstraint{}"
		}
	} else if strings.HasSuffix(p.Name, "initializer") && p.ObjectName != "LRandomFourierFeatures" {
		if stringValue != "nil" {
			if config, ok := p.Default.(map[string]interface{}); ok {
				stringValue = fmt.Sprintf("initializer.%s()", config["class_name"])
			} else {
				stringValue = fmt.Sprintf("initializer.%s()", strings.Title(snakeCaseToCamelCase(fmt.Sprint(p.Default))))
			}
		} else {
			stringValue = "&initializer.NilInitializer{}"
		}
	} else if strings.HasSuffix(p.Name, "regularizer") {
		if stringValue != "nil" {
			if config, ok := p.Default.(map[string]interface{}); ok {
				stringValue = fmt.Sprintf("regularizer.%s()", config["class_name"])
			} else {
				stringValue = fmt.Sprintf("regularizer.%s()", strings.Title(snakeCaseToCamelCase(fmt.Sprint(p.Default))))
			}
		} else {
			stringValue = "&regularizer.NilRegularizer{}"
		}
	} else if p.Name == "dtype" && stringValue == "nil" {
		stringValue = "Float32"
	} else if p.Name == "dtype" {
		stringValue = strings.Title(strings.ReplaceAll(stringValue, `"`, ``))
	} else if p.Name == "trainable" {
		stringValue = "true"
	} else if p.Name == "name" {
		stringValue = fmt.Sprintf("UniqueName(\"%s\")", strings.ReplaceAll(stringValue, `"`, ``))
	}

	return stringValue
}

type fileGenerator struct {
	Dir    string
	Params []*parameter
	object objectJson
}

func newFileGenerator(object objectJson, dir string, additionalParams ...*parameter) *fileGenerator {
	f := &fileGenerator{
		Dir:    dir,
		object: object,
	}
	f.loadParams()
	for _, param := range additionalParams {
		param.ObjectName = fmt.Sprintf("%s%s", strings.Title(string(object.Type[0])), object.Name)
		param.IsAdditional = true
		f.addParam(param)
	}
	sort.SliceStable(f.Params, func(i, j int) bool {
		return f.Params[i].Name < f.Params[j].Name
	})
	return f
}

func (f *fileGenerator) loadParams() {
	for _, param := range f.object.RequiredParams {
		paramSliceInterface, ok := param.([]interface{})
		if ok {
			f.addParam(&parameter{
				ObjectName: fmt.Sprintf("%s%s", strings.Title(string(f.object.Type[0])), f.object.Name),
				Name:       paramSliceInterface[0].(string),
				Default:    paramSliceInterface[1],
				IsRequired: true,
			})
		}
	}
	for _, param := range f.object.OptionalParams {
		paramSliceInterface, ok := param.([]interface{})
		if ok {
			f.addParam(&parameter{
				ObjectName: fmt.Sprintf("%s%s", strings.Title(string(f.object.Type[0])), f.object.Name),
				Name:       paramSliceInterface[0].(string),
				Default:    paramSliceInterface[1],
				IsRequired: false,
			})
		}
	}
	for key, defaultValue := range f.object.Config["config"].(map[string]interface{}) {
		f.addParam(&parameter{
			ObjectName: fmt.Sprintf("%s%s", strings.Title(string(f.object.Type[0])), f.object.Name),
			Name:       key,
			Default:    defaultValue,
			IsRequired: false,
		})
	}
}

func (f *fileGenerator) addParam(param *parameter) {
	found := false
	for _, p := range f.Params {
		if p.Name == param.Name {
			found = true
		}
	}
	if !found {
		f.Params = append(f.Params, param)
	}
}

func (f *fileGenerator) getOptionString(param *parameter) string {
	receiver := strings.ToLower(string(param.ObjectName[0]))
	lines := []string{
		fmt.Sprintf(
			"func (%s *%s) Set%s(%s %s) *%s {",
			receiver,
			param.ObjectName,
			strings.Title(param.getCamelCaseName()),
			param.getCamelCaseName(),
			param.getGolangType(),
			param.ObjectName,
		),
		fmt.Sprintf(
			"\t %s.%s = %s",
			receiver,
			param.getCamelCaseName(),
			param.getCamelCaseName(),
		),
		fmt.Sprintf(
			"\treturn %s",
			receiver,
		),
		"}",
	}
	return strings.Join(lines, "\n")
}

func (f *fileGenerator) generate() {
	e := os.MkdirAll(f.Dir, os.ModePerm)
	if e != nil {
		panic(e)
	}

	reciever := strings.ToLower(string(f.object.Type[0]))
	structName := fmt.Sprintf("%s%s", strings.Title(string(f.object.Type[0])), f.object.Name)
	var options []string
	var objectProperties []string
	var paramSetters []string
	var constructorParamsString string

	configLines := []string{
		"map[string]interface{}{",
	}
	for _, param := range f.Params {
		objectProperties = append(objectProperties, fmt.Sprintf("%s %s", param.getCamelCaseName(), param.getGolangType()))
		if param.IsRequired {
			constructorParamsString += fmt.Sprintf(
				"%s %s, ",
				param.getCamelCaseName(),
				param.getGolangType(),
			)
			paramSetters = append(paramSetters, fmt.Sprintf("%s: %s,", param.getCamelCaseName(), param.getCamelCaseName()))
		} else {
			if param.Name != "inputs" {
				options = append(options, f.getOptionString(param))
			}
			if !param.IsAdditional {
				paramSetters = append(paramSetters, fmt.Sprintf("%s: %s,", param.getCamelCaseName(), param.getStringDefaultValue()))
			}
		}

		if !param.IsAdditional {
			getter := param.getCamelCaseName()
			if param.Name == "dtype" {
				getter = "dtype.String()"
			}
			if strings.HasSuffix(param.Name, "constraint") ||
				(strings.HasSuffix(param.Name, "initializer") && structName != "LRandomFourierFeatures") ||
				strings.HasSuffix(param.Name, "regularizer") {
				getter += ".GetKerasLayerConfig()"
			}
			configLines = append(configLines, fmt.Sprintf(
				"\t\"%s\": %s.%s,",
				param.Name,
				reciever,
				getter,
			))
		}
	}
	configLines = append(configLines, "}")
	layerDefaultGetters := ""
	if f.object.Type == "layer" {
		layerDefaultGetters = fmt.Sprintf(
			`
func (%s *%s) GetShape() tf.Shape {
	return %s.shape
}

func (%s *%s) GetDtype() DataType {
	return %s.dtype
}

func (%s *%s) SetInputs(inputs ...Layer) Layer {
	%s.inputs = inputs
	return %s
}

func (%s *%s) GetInputs() []Layer {
	return %s.inputs
}

func (%s *%s) GetName() string {
	return %s.name
}
`,
			reciever,
			structName,
			reciever,
			reciever,
			structName,
			reciever,
			reciever,
			structName,
			reciever,
			reciever,
			reciever,
			structName,
			reciever,
			reciever,
			structName,
			reciever,
		)
	}

	inboundNodes := ""
	if f.object.Type == "layer" {
		inboundNodes = fmt.Sprintf(`	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range %s.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}`, reciever)
	}
	inboundNodesSetter := ""
	configInboundNodesDef := ""
	if f.object.Type == "layer" {
		inboundNodesSetter = "\t\tInboundNodes: inboundNodes,"
		configInboundNodesDef = "\tInboundNodes [][][]interface{} `json:\"inbound_nodes\"`"
	}
	lines := []string{
		fmt.Sprintf("type %s struct {", structName),
		"\t" + strings.Join(objectProperties, "\n\t"),
		"}",
		"",
		fmt.Sprintf("func %s(%s) *%s {", f.object.Name, constructorParamsString, structName),
		fmt.Sprintf(
			"\treturn &%s{\n\t\t%s\t\n\t}",
			structName,
			strings.Join(paramSetters, "\n\t\t\t"),
		),
		"}",
		"",
		strings.Join(options, "\n\n"),
		"",
		layerDefaultGetters,
		"",
		fmt.Sprintf("type jsonConfig%s struct {", structName),
		"\tClassName string `json:\"class_name\"`",
		"\tName string `json:\"name\"`",
		"\tConfig map[string]interface{} `json:\"config\"`",
		configInboundNodesDef,
		"}",
		fmt.Sprintf(
			"func (%s *%s) GetKerasLayerConfig() interface{} {",
			reciever,
			structName,
		),
		inboundNodes,
		fmt.Sprintf("\treturn jsonConfig%s{", structName),
		fmt.Sprintf("\t\tClassName: \"%s\",", f.object.Config["class_name"]),
		fmt.Sprintf("\t\tName: %s.name,", reciever),
		fmt.Sprintf("\t\tConfig: %s,", strings.Join(configLines, "\n")),
		inboundNodesSetter,
		"\t}",
		"}",
		"",
		fmt.Sprintf("func (%s *%s) GetCustomLayerDefinition() string {", reciever, structName),
		"\treturn ``",
		"}",
		"",
	}

	importedLines := []string{
		"package " + f.object.Type,
		"",
	}

	constraintAdded, initializerAdded, regularizerAdded, tfAdded := false, false, false, false

	for _, line := range lines {
		if strings.Contains(line, "constraint.") {
			if !constraintAdded {
				importedLines = append(importedLines, "import \"github.com/codingbeard/tfkg/layer/constraint\"")
				constraintAdded = true
			}
		}
		if strings.Contains(line, "initializer.") {
			if !initializerAdded {
				importedLines = append(importedLines, "import \"github.com/codingbeard/tfkg/layer/initializer\"")
				initializerAdded = true
			}
		}
		if strings.Contains(line, "regularizer.") {
			if !regularizerAdded {
				importedLines = append(importedLines, "import \"github.com/codingbeard/tfkg/layer/regularizer\"")
				regularizerAdded = true
			}
		}
		if strings.Contains(line, "tf.") {
			if !tfAdded {
				importedLines = append(importedLines, "import tf \"github.com/galeone/tensorflow/tensorflow/go\"")
				tfAdded = true
			}
		}
	}
	importedLines = append(importedLines, "")

	for _, line := range lines {
		importedLines = append(importedLines, line)
	}

	e = ioutil.WriteFile(
		filepath.Join(f.Dir, fmt.Sprintf("%s.go", f.object.Name)),
		[]byte(strings.Join(importedLines, "\n")),
		os.ModePerm,
	)
	if e != nil {
		panic(e)
	}
}

func snakeCaseToCamelCase(str string) string {
	str = strings.ReplaceAll(str, "_", " ")
	str = strings.Title(str)
	str = strings.ReplaceAll(str, " ", "")
	return strings.ToLower(string(str[0])) + str[1:]
}

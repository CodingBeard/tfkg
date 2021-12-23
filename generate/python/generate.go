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
		if object.Type == "initializer" {
			createInitializer(object)
		} else if object.Type == "regularizer" {
			createRegularizer(object)
		} else if object.Type == "constraint" {
			createConstraint(object)
		} else if object.Type == "layer" {
			createLayer(object)
		}
	}

	_, e = exec.Command("go", "fmt", "github.com/codingbeard/tfkg/layer").Output()
	if e != nil {
		panic(e)
	}
}

func createInitializer(object objectJson) {
	e := os.MkdirAll("../../layer/initializer", os.ModePerm)
	if e != nil {
		panic(e)
	}

	var setters []string
	var objectProperties []string
	for _, param := range getRequiredParams(object) {
		objectProperties = append(objectProperties, fmt.Sprintf("%s %s", param[0], param[1]))
	}
	for _, param := range getOptionalParams(object) {
		objectProperties = append(objectProperties, fmt.Sprintf("%s %s", param[0], param[1]))
		setters = append(setters, getOptionString(object.Name, param[0], param[1]))
	}

	var requiredParamSetters []string
	for _, paramName := range getRequiredParamNames(object) {
		requiredParamSetters = append(requiredParamSetters, fmt.Sprintf("%s: %s", paramName, paramName))
	}

	var defaultParamSetters []string
	for _, param := range getOptionalParamDefaults(object) {
		defaultParamSetters = append(defaultParamSetters, fmt.Sprintf("%s: %s,", param[0], param[1]))
	}

	lines := []string{
		"package initializer",
		"",
		fmt.Sprintf("type %s struct {", object.Name),
		"\t" + strings.Join(objectProperties, "\n\t"),
		"}",
		"",
		fmt.Sprintf("func New%s(%s) *%s {", object.Name, getRequiredParamsString(object), object.Name),
		fmt.Sprintf(
			"\treturn &%s{\n\t\t%s%s\t\n\t}",
			object.Name,
			strings.Join(requiredParamSetters, "\n\t\t"),
			strings.Join(defaultParamSetters, "\n\t\t"),
		),
		"}",
		"",
		strings.Join(setters, "\n\n"),
		"",
		fmt.Sprintf("type jsonConfig%s struct {", object.Name),
		"\tClassName string `json:\"class_name\"`",
		"\tName string `json:\"name\"`",
		"\tConfig map[string]interface{} `json:\"config\"`",
		"}",
		fmt.Sprintf(
			"func (%s *%s) GetKerasLayerConfig() interface{} {",
			strings.ToLower(string(object.Name[0])),
			object.Name,
		),
		fmt.Sprintf("\tif %s == nil {", strings.ToLower(string(object.Name[0]))),
		"\t\treturn nil",
		"\t}",
		fmt.Sprintf("\treturn jsonConfig%s{", object.Name),
		fmt.Sprintf("\t\tClassName: \"%s\",", object.Config["class_name"]),
		fmt.Sprintf("\t\tConfig: %s,", getConfigValue(object)),
		"\t}",
		"}",
	}

	e = ioutil.WriteFile(
		filepath.Join("../../layer/initializer", fmt.Sprintf("%s.go", object.Name)),
		[]byte(strings.Join(lines, "\n")),
		os.ModePerm,
	)
	if e != nil {
		panic(e)
	}
}

func createRegularizer(object objectJson) {
	e := os.MkdirAll("../../layer/regularizer", os.ModePerm)
	if e != nil {
		panic(e)
	}

	var setters []string
	var objectProperties []string
	for _, param := range getRequiredParams(object) {
		objectProperties = append(objectProperties, fmt.Sprintf("%s %s", param[0], param[1]))
	}
	for _, param := range getOptionalParams(object) {
		objectProperties = append(objectProperties, fmt.Sprintf("%s %s", param[0], param[1]))
		setters = append(setters, getOptionString(object.Name, param[0], param[1]))
	}

	var requiredParamSetters []string
	for _, paramName := range getRequiredParamNames(object) {
		requiredParamSetters = append(requiredParamSetters, fmt.Sprintf("%s: %s", paramName, paramName))
	}

	var defaultParamSetters []string
	for _, param := range getOptionalParamDefaults(object) {
		defaultParamSetters = append(defaultParamSetters, fmt.Sprintf("%s: %s,", param[0], param[1]))
	}

	lines := []string{
		"package regularizer",
		"",
		fmt.Sprintf("type %s struct {", object.Name),
		"\t" + strings.Join(objectProperties, "\n\t"),
		"}",
		"",
		fmt.Sprintf("func New%s(%s) *%s {", object.Name, getRequiredParamsString(object), object.Name),
		fmt.Sprintf(
			"\treturn &%s{\n\t\t%s%s\t\n\t}",
			object.Name,
			strings.Join(requiredParamSetters, "\n\t\t"),
			strings.Join(defaultParamSetters, "\n\t\t"),
		),
		"}",
		"",
		strings.Join(setters, "\n\n"),
		"",
		fmt.Sprintf("type jsonConfig%s struct {", object.Name),
		"\tClassName string `json:\"class_name\"`",
		"\tName string `json:\"name\"`",
		"\tConfig map[string]interface{} `json:\"config\"`",
		"}",
		fmt.Sprintf(
			"func (%s *%s) GetKerasLayerConfig() interface{} {",
			strings.ToLower(string(object.Name[0])),
			object.Name,
		),
		fmt.Sprintf("\tif %s == nil {", strings.ToLower(string(object.Name[0]))),
		"\t\treturn nil",
		"\t}",
		fmt.Sprintf("\treturn jsonConfig%s{", object.Name),
		fmt.Sprintf("\t\tClassName: \"%s\",", object.Config["class_name"]),
		fmt.Sprintf("\t\tConfig: %s,", getConfigValue(object)),
		"\t}",
		"}",
	}

	e = ioutil.WriteFile(
		filepath.Join("../../layer/regularizer", fmt.Sprintf("%s.go", object.Name)),
		[]byte(strings.Join(lines, "\n")),
		os.ModePerm,
	)
	if e != nil {
		panic(e)
	}
}

func createConstraint(object objectJson) {
	e := os.MkdirAll("../../layer/constraint", os.ModePerm)
	if e != nil {
		panic(e)
	}

	var setters []string
	var objectProperties []string
	for _, param := range getRequiredParams(object) {
		objectProperties = append(objectProperties, fmt.Sprintf("%s %s", param[0], param[1]))
	}
	for _, param := range getOptionalParams(object) {
		objectProperties = append(objectProperties, fmt.Sprintf("%s %s", param[0], param[1]))
		setters = append(setters, getOptionString(object.Name, param[0], param[1]))
	}

	var requiredParamSetters []string
	for _, paramName := range getRequiredParamNames(object) {
		requiredParamSetters = append(requiredParamSetters, fmt.Sprintf("%s: %s", paramName, paramName))
	}

	var defaultParamSetters []string
	for _, param := range getOptionalParamDefaults(object) {
		defaultParamSetters = append(defaultParamSetters, fmt.Sprintf("%s: %s,", param[0], param[1]))
	}

	lines := []string{
		"package constraint",
		"",
		fmt.Sprintf("type %s struct {", object.Name),
		"\t" + strings.Join(objectProperties, "\n\t"),
		"}",
		"",
		fmt.Sprintf("func New%s(%s) *%s {", object.Name, getRequiredParamsString(object), object.Name),
		fmt.Sprintf(
			"\treturn &%s{\n\t\t%s%s\t\n\t}",
			object.Name,
			strings.Join(requiredParamSetters, "\n\t\t"),
			strings.Join(defaultParamSetters, "\n\t\t"),
		),
		"}",
		"",
		strings.Join(setters, "\n\n"),
		"",
		fmt.Sprintf("type jsonConfig%s struct {", object.Name),
		"\tClassName string `json:\"class_name\"`",
		"\tName string `json:\"name\"`",
		"\tConfig map[string]interface{} `json:\"config\"`",
		"}",
		fmt.Sprintf(
			"func (%s *%s) GetKerasLayerConfig() interface{} {",
			strings.ToLower(string(object.Name[0])),
			object.Name,
		),
		fmt.Sprintf("\tif %s == nil {", strings.ToLower(string(object.Name[0]))),
		"\t\treturn nil",
		"\t}",
		fmt.Sprintf("\treturn jsonConfig%s{", object.Name),
		fmt.Sprintf("\t\tClassName: \"%s\",", object.Config["class_name"]),
		fmt.Sprintf("\t\tConfig: %s,", getConfigValue(object)),
		"\t}",
		"}",
	}

	e = ioutil.WriteFile(
		filepath.Join("../../layer/constraint", fmt.Sprintf("%s.go", object.Name)),
		[]byte(strings.Join(lines, "\n")),
		os.ModePerm,
	)
	if e != nil {
		panic(e)
	}
}

func createLayer(object objectJson) {
	e := os.MkdirAll("../../layer", os.ModePerm)
	if e != nil {
		panic(e)
	}

	var setters []string
	var objectProperties []string
	objectPropertyNames := make(map[string]bool)
	for _, param := range getRequiredParams(object) {
		objectProperties = append(objectProperties, fmt.Sprintf("%s %s", param[0], param[1]))
		objectPropertyNames[param[0]] = true
	}
	setters = append(setters, getOptionString(object.Name, "name", "string"))
	setters = append(setters, getOptionString(object.Name, "dtype", "DataType"))
	setters = append(setters, getOptionString(object.Name, "trainable", "bool"))
	for _, param := range getOptionalParams(object) {
		if param[0] != "dtype" && param[0] != "name" && param[0] != "trainable" {
			objectProperties = append(objectProperties, fmt.Sprintf("%s %s", param[0], param[1]))
			objectPropertyNames[param[0]] = true
		}
		if param[0] != "name" && param[0] != "trainable" && param[0] != "dtype" {
			setters = append(setters, getOptionString(object.Name, param[0], param[1]))
		}
	}

	subConfig := object.Config["config"].(map[string]interface{})
	for originalName, value := range subConfig {
		name := snakeCaseToCamelCase(originalName)
		if _, ok := objectPropertyNames[name]; ok {
			continue
		}
		if name != "dtype" && name != "name" && name != "trainable" {
			objectProperties = append(objectProperties, fmt.Sprintf("%s %s", snakeCaseToCamelCase(name), getGolangTypeFromValue(object.Name, originalName, value)))
		}
	}

	var requiredParamSetters []string
	objectParamSetterNames := make(map[string]bool)
	for _, paramName := range getRequiredParamNames(object) {
		if paramName == "trainable" {
			continue
		}
		requiredParamSetters = append(requiredParamSetters, fmt.Sprintf("%s: %s,", paramName, paramName))
		objectParamSetterNames[paramName] = true
	}

	var defaultParamSetters []string
	for _, param := range getOptionalParamDefaults(object) {
		if param[0] == "name" || param[0] == "trainable" {
			continue
		}
		defaultParamSetters = append(defaultParamSetters, fmt.Sprintf("%s: %s,", param[0], param[1]))
		objectParamSetterNames[param[0]] = true
	}
	for originalName, value := range subConfig {
		name := snakeCaseToCamelCase(originalName)
		if _, ok := objectParamSetterNames[name]; ok {
			continue
		}
		if name != "dtype" && name != "name" && name != "trainable" {
			defaultParamSetters = append(defaultParamSetters, fmt.Sprintf("%s: %s,", snakeCaseToCamelCase(name), getGolangStringFromValue(object.Name, originalName, value)))
		}
	}
	defaultParamSetters = append(defaultParamSetters, "trainable: true,")
	defaultParamSetters = append(defaultParamSetters, "inputs: inputs,")
	defaultParamSetters = append(defaultParamSetters, fmt.Sprintf("name: UniqueName(\"%s\"),", strings.ToLower(object.Name)))

	requiredParamSettersString := strings.Join(requiredParamSetters, "\n\t\t\t")
	if len(requiredParamSetters) > 0 {
		requiredParamSettersString += "\n\t\t\t"
	}
	reciever := strings.ToLower(string(object.Name[0]))
	lines := []string{
		fmt.Sprintf("type %s struct {", object.Name),
		"\tname string",
		"\tdtype DataType",
		"\tinputs []Layer",
		"\tshape tf.Shape",
		"\ttrainable bool",
		"\t" + strings.Join(objectProperties, "\n\t"),
		"}",
		"",
		fmt.Sprintf(
			"func New%s(%soptions ...%sOption) func(inputs ...Layer) Layer {",
			object.Name,
			getRequiredParamsString(object),
			object.Name,
		),
		"\treturn func(inputs ...Layer) Layer {",
		fmt.Sprintf(
			"\t\t%s := &%s{\n\t\t\t%s%s\t\t\n\t\t}",
			reciever,
			object.Name,
			requiredParamSettersString,
			strings.Join(defaultParamSetters, "\n\t\t\t"),
		),
		"\t\tfor _, option := range options {",
		fmt.Sprintf("\t\t\toption(%s)", reciever),
		"\t\t}",
		fmt.Sprintf("\t\treturn %s", reciever),
		"\t}",
		"}",
		"",
		fmt.Sprintf("type %sOption func (*%s)", object.Name, object.Name),
		"",
		strings.Join(setters, "\n\n"),
		"",
		fmt.Sprintf(
			`
func (%s *%s) GetShape() tf.Shape {
	return %s.shape
}

func (%s *%s) GetDtype() DataType {
	return %s.dtype
}

func (%s *%s) SetInput(inputs []Layer) {
	%s.inputs = inputs
	%s.dtype = inputs[0].GetDtype()
}

func (%s *%s) GetInputs() []Layer {
	return %s.inputs
}

func (%s *%s) GetName() string {
	return %s.name
}
`,
			reciever,
			object.Name,
			reciever,
			reciever,
			object.Name,
			reciever,
			reciever,
			object.Name,
			reciever,
			reciever,
			reciever,
			object.Name,
			reciever,
			reciever,
			object.Name,
			reciever,
		),
		"",
		fmt.Sprintf("type jsonConfig%s struct {", object.Name),
		"\tClassName string `json:\"class_name\"`",
		"\tName string `json:\"name\"`",
		"\tConfig map[string]interface{} `json:\"config\"`",
		"\tInboundNodes [][][]interface{} `json:\"inbound_nodes\"`",
		"}",
		fmt.Sprintf(
			"func (%s *%s) GetKerasLayerConfig() interface{} {",
			strings.ToLower(string(object.Name[0])),
			object.Name,
		),
		fmt.Sprintf(`	inboundNodes := [][][]interface{}{
		{},
	}
	for _, input := range %s.inputs {
		inboundNodes[0] = append(inboundNodes[0], []interface{}{
			input.GetName(),
			0,
			0,
			map[string]bool{},
		})
	}`, reciever),
		fmt.Sprintf("\treturn jsonConfig%s{", object.Name),
		fmt.Sprintf("\t\tClassName: \"%s\",", object.Config["class_name"]),
		fmt.Sprintf("\t\tName: %s.name,", reciever),
		fmt.Sprintf("\t\tConfig: %s,", getConfigValue(object)),
		"\t\tInboundNodes: inboundNodes,",
		"\t}",
		"}",
		"",
		fmt.Sprintf("func (%s *%s) GetCustomLayerDefinition() string {", reciever, object.Name),
		"\treturn ``",
		"}",
		"",
	}

	importedLines := []string{
		"package layer",
		"",
		"import tf \"github.com/galeone/tensorflow/tensorflow/go\"",
	}

	constraintAdded, initializerAdded, regularizerAdded := false, false, false

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
	}
	importedLines = append(importedLines, "")

	for _, line := range lines {
		importedLines = append(importedLines, line)
	}

	e = ioutil.WriteFile(
		filepath.Join("../../layer", fmt.Sprintf("%s.go", object.Name)),
		[]byte(strings.Join(importedLines, "\n")),
		os.ModePerm,
	)
	if e != nil {
		panic(e)
	}
}

func getConfigValue(object objectJson) string {
	lines := []string{
		"map[string]interface{}{",
	}
	reciever := strings.ToLower(string(object.Name[0]))

	subConfig := object.Config["config"].(map[string]interface{})
	var sortedConfigKeys []string
	for name := range subConfig {
		sortedConfigKeys = append(sortedConfigKeys, name)
	}
	sort.Strings(sortedConfigKeys)
	for _, name := range sortedConfigKeys {
		getter := snakeCaseToCamelCase(name)
		if name == "dtype" {
			getter = "dtype.String()"
		}
		if strings.HasSuffix(name, "constraint") || (strings.HasSuffix(name, "initializer") && object.Name != "RandomFourierFeatures") || strings.HasSuffix(name, "regularizer") {
			getter += ".GetKerasLayerConfig()"
		}
		lines = append(lines, fmt.Sprintf(
			"\t\"%s\": %s.%s,",
			name,
			reciever,
			getter,
		))
	}

	lines = append(lines, "}")

	return strings.Join(lines, "\n\t\t")
}

func getOptionString(objectName string, name string, typeString string) string {
	lines := []string{
		fmt.Sprintf(
			"func %sWith%s(%s %s) func(%s *%s) {",
			objectName,
			strings.Title(name),
			name,
			typeString,
			strings.ToLower(string(objectName[0])),
			objectName,
		),
		fmt.Sprintf(
			"\t return func(%s *%s) {",
			strings.ToLower(string(objectName[0])),
			objectName,
		),
		fmt.Sprintf("\t\t%s.%s = %s", strings.ToLower(string(objectName[0])), name, name),
		"\t}",
		"}",
	}
	return strings.Join(lines, "\n")
}

func getRequiredParamsString(object objectJson) string {
	var params []string

	for _, paramInterface := range object.RequiredParams {
		paramSliceInterface, ok := paramInterface.([]interface{})
		if ok && len(paramSliceInterface) == 2 {
			params = append(params, fmt.Sprintf(
				"%s %s",
				snakeCaseToCamelCase(paramSliceInterface[0].(string)),
				getGolangTypeFromValue(object.Name, paramSliceInterface[0].(string), paramSliceInterface[1]),
			))
		}
	}

	if len(params) > 0 {
		return fmt.Sprintf("%s", strings.Join(params, ", ")) + ", "
	} else {
		return ""
	}
}

func getRequiredParams(object objectJson) [][]string {
	var params [][]string

	for _, paramInterface := range object.RequiredParams {
		paramSliceInterface, ok := paramInterface.([]interface{})
		if ok {
			params = append(params, []string{
				snakeCaseToCamelCase(paramSliceInterface[0].(string)),
				getGolangTypeFromValue(object.Name, paramSliceInterface[0].(string), paramSliceInterface[1]),
			})
		} else {
			fmt.Println(fmt.Sprintf("%T", paramInterface))
			os.Exit(1)
		}
	}

	return params
}

func getOptionalParams(object objectJson) [][]string {
	var params [][]string

	for _, paramInterface := range object.OptionalParams {
		paramSliceInterface, ok := paramInterface.([]interface{})
		if ok {
			params = append(params, []string{
				snakeCaseToCamelCase(paramSliceInterface[0].(string)),
				getGolangTypeFromValue(object.Name, paramSliceInterface[0].(string), paramSliceInterface[1]),
			})
		} else {
			fmt.Println(fmt.Sprintf("%T", paramInterface))
			os.Exit(1)
		}
	}

	return params
}

func getOptionalParamDefaults(object objectJson) [][]string {
	var params [][]string

	for _, paramInterface := range object.OptionalParams {
		paramSliceInterface, ok := paramInterface.([]interface{})
		if ok {
			params = append(params, []string{
				snakeCaseToCamelCase(paramSliceInterface[0].(string)),
				getGolangStringFromValue(object.Name, paramSliceInterface[0].(string), paramSliceInterface[1]),
			})
		} else {
			fmt.Println(fmt.Sprintf("%T", paramInterface))
			os.Exit(1)
		}
	}

	return params
}

func getRequiredParamNames(object objectJson) []string {
	var params []string

	for _, paramInterface := range object.RequiredParams {
		paramSliceInterface, ok := paramInterface.([]interface{})
		if ok && len(paramSliceInterface) == 2 {
			params = append(params, snakeCaseToCamelCase(paramSliceInterface[0].(string)))
		}
	}

	return params
}

func snakeCaseToCamelCase(str string) string {
	str = strings.ReplaceAll(str, "_", " ")
	str = strings.Title(str)
	str = strings.ReplaceAll(str, " ", "")
	return strings.ToLower(string(str[0])) + str[1:]
}

func getGolangTypeFromValue(objectName string, paramName string, value interface{}) string {
	stringValue := ""
	if strings.HasSuffix(paramName, "constraint") {
		stringValue = "constraint.Constraint"
	} else if strings.HasSuffix(paramName, "initializer") && objectName != "RandomFourierFeatures" {
		stringValue = "initializer.Initializer"
	} else if strings.HasSuffix(paramName, "regularizer") {
		stringValue = "regularizer.Regularizer"
	} else if strings.HasSuffix(paramName, "activation") {
		stringValue = "string"
	} else {
		stringValue = fmt.Sprintf("%T", value)
		if stringValue == "<nil>" {
			stringValue = "interface{}"
		}
		if paramName == "dtype" {
			stringValue = "DataType"
		}
		if paramName == "name" {
			stringValue = "string"
		}
	}
	return stringValue
}

func getGolangStringFromValue(objectName string, paramName string, value interface{}) string {
	stringValue := fmt.Sprintf("%#v", value)
	if stringValue == "<nil>" {
		stringValue = "nil"
	}
	if paramName == "activation" && stringValue == "nil" {
		stringValue = `"linear"`
	} else if paramName == "embeddings_initializer" && fmt.Sprint(value) == "uniform" {
		stringValue = "&initializer.RandomUniform{}"
	} else if strings.HasSuffix(paramName, "constraint") {
		if stringValue != "nil" {
			if config, ok := value.(map[string]interface{}); ok {
				stringValue = fmt.Sprintf("&constraint.%s{}", config["class_name"])
			} else {
				stringValue = fmt.Sprintf("&constraint.%s{}", strings.Title(snakeCaseToCamelCase(fmt.Sprint(value))))
			}
		} else {
			stringValue = "&constraint.NilConstraint{}"
		}
	} else if strings.HasSuffix(paramName, "initializer") && objectName != "RandomFourierFeatures" {
		if stringValue != "nil" {
			if config, ok := value.(map[string]interface{}); ok {
				stringValue = fmt.Sprintf("&initializer.%s{}", config["class_name"])
			} else {
				stringValue = fmt.Sprintf("&initializer.%s{}", strings.Title(snakeCaseToCamelCase(fmt.Sprint(value))))
			}
		} else {
			stringValue = "&initializer.NilInitializer{}"
		}
	} else if strings.HasSuffix(paramName, "regularizer") {
		if stringValue != "nil" {
			if config, ok := value.(map[string]interface{}); ok {
				stringValue = fmt.Sprintf("&regularizer.%s{}", config["class_name"])
			} else {
				stringValue = fmt.Sprintf("&regularizer.%s{}", strings.Title(snakeCaseToCamelCase(fmt.Sprint(value))))
			}
		} else {
			stringValue = "&regularizer.NilRegularizer{}"
		}
	} else if paramName == "dtype" && stringValue == "nil" {
		stringValue = "Float32"
	} else if paramName == "trainable" {
		stringValue = "true"
	}

	return stringValue
}

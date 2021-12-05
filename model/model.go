package model

import (
	"fmt"
	tf "github.com/codingbeard/tfkg/tensorflow/go"
	"github.com/codingbeard/tfkg/tensorflow/go/core/framework/node_def_go_proto"
	"strings"
)

type Model struct {
	model *tf.SavedModel
}

func Load(modelDir string, tags []string, options *tf.SessionOptions) (*Model, error) {
	m, e := tf.LoadSavedModel(modelDir, tags, options)
	return &Model{model: m}, e
}

func (m *Model) Raw() *tf.SavedModel {
	return m.model
}

func (m *Model) getOpCode(nodeDef *node_def_go_proto.NodeDef) string {
	return ""
}

type definedVariable struct {
	name       string
	definition string
}

func cleanVariableName(name string) string {
	return strings.Split(strings.ReplaceAll(name, "^", ""), ":")[0]
}

func (m *Model) defineVariable(
	nodeDef *node_def_go_proto.NodeDef,
	nodeDefs map[string]*node_def_go_proto.NodeDef,
	definedVariables []definedVariable,
	variableNames map[string]string,
) (map[string]string, []definedVariable) {
	if _, ok := variableNames[cleanVariableName(nodeDef.Name)]; ok {
		return variableNames, definedVariables
	}
	variableName := strings.ToLower(strings.ReplaceAll(cleanVariableName(nodeDef.Name), "/", "_"))
	if len(nodeDef.Input) == 0 {
		variableNames[cleanVariableName(nodeDef.Name)] = variableName
		definedVariables = append(definedVariables, definedVariable{
			name: variableName,
			definition: fmt.Sprintf(
				`op.%s(scope.SubScope("%s"))`,
				nodeDef.Op,
				nodeDef.Name,
			),
		})
	} else {
		var variableInputs []string
		for _, inputName := range nodeDef.Input {
			inputName = cleanVariableName(inputName)
			finalVariableName, ok := variableNames[inputName]
			if !ok {
				def, ok := nodeDefs[inputName]
				if ok {
					variableNames, definedVariables = m.defineVariable(def, nodeDefs, definedVariables, variableNames)
				}
				if _, ok := variableNames[inputName]; ok {
					finalVariableName = variableNames[inputName]
				} else if strings.HasSuffix(inputName, "_resource") {
					variableNames[inputName] = inputName
					definedVariables = append(definedVariables, definedVariable{
						name: inputName,
						definition: fmt.Sprintf(
							`op.VarHandleOp(scope.SubScope("%s"))`,
							inputName,
						),
					})
					finalVariableName = inputName
				} else {
					finalVariableName = fmt.Sprintf(`"%s"`, inputName)
				}
			}
			variableInputs = append(variableInputs, finalVariableName)
		}

		variableNames[cleanVariableName(nodeDef.Name)] = variableName
		definedVariables = append(definedVariables, definedVariable{
			name: variableName,
			definition: fmt.Sprintf(
				`op.%s(scope.SubScope("%s"), %s)`,
				nodeDef.Op,
				nodeDef.Name,
				strings.Join(variableInputs, ", "),
			),
		})
	}

	return variableNames, definedVariables
}

func (m *Model) DecompileGraphToGolangCode() (string, error) {
	var lines []string

	for signatureName, signatureDef := range m.Raw().GraphDef.SignatureDef {
		_, _ = signatureName, signatureDef
	}

	nodeDefs := make(map[string]*node_def_go_proto.NodeDef)
	variableNames := make(map[string]string)
	var definedVariables []definedVariable

	for _, nodeDef := range m.Raw().GraphDef.GraphDef.GetNode() {
		if strings.HasPrefix(nodeDef.Name, "StatefulPartitionedCall") {
			continue
		}
		nodeDefs[cleanVariableName(nodeDef.Name)] = nodeDef
	}

	for _, nodeDef := range m.Raw().GraphDef.GraphDef.GetNode() {
		if strings.HasPrefix(nodeDef.Name, "StatefulPartitionedCall") {
			continue
		}
		variableNames, definedVariables = m.defineVariable(nodeDef, nodeDefs, definedVariables, variableNames)
	}

	for _, function := range m.Raw().GraphDef.GraphDef.Library.Function {
		_ = function
		if strings.Contains(function.Signature.Name, "learn") {
			for _, nodeDef := range function.NodeDef {
				nodeDefs[cleanVariableName(nodeDef.Name)] = nodeDef
			}

			for _, nodeDef := range function.NodeDef {
				variableNames, definedVariables = m.defineVariable(nodeDef, nodeDefs, definedVariables, variableNames)
			}

			for _, variable := range definedVariables {
				for _, varName := range variableNames {
					variable.definition = strings.ReplaceAll(variable.definition, fmt.Sprintf("{Unknown variable: %s}", varName+"_resource"), varName)
				}
				lines = append(lines, fmt.Sprintf("%s = %s", variable.name, variable.definition))
			}
		}
	}

	return strings.Join(lines, "\n"), nil

	for _, variable := range definedVariables {
		lines = append(lines, fmt.Sprintf("%s = %s", variable.name, variable.definition))
	}

	return strings.Join(lines, "\n"), nil
}

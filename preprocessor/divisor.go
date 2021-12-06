package preprocessor

import (
	"encoding/json"
	"fmt"
	"github.com/codingbeard/cberrors"
	"io/ioutil"
	"os"
	"strconv"
)

type RegressionDivisor struct {
	divisors []float32

	errorHandler *cberrors.ErrorsContainer
}

type divisorConfig map[string]float32

func NewDivisor(
	errorHandler *cberrors.ErrorsContainer,
) *RegressionDivisor {
	return &RegressionDivisor{
		errorHandler: errorHandler,
	}
}

func (r *RegressionDivisor) Load(configFile string) error {
	contents, e := ioutil.ReadFile(configFile)
	if e != nil {
		return e
	}

	var config divisorConfig

	e = json.Unmarshal(contents, &config)
	if e != nil {
		return e
	}

	for i := 0; i < len(config); i++ {
		key := strconv.Itoa(i)

		divisor, ok := config[key]
		if !ok {
			return fmt.Errorf("missing divisor: %s in configFile: %s", key, configFile)
		}
		r.divisors = append(r.divisors, divisor)
	}

	return nil
}

func (r *RegressionDivisor) Save(configFile string) error {
	divisors := make(map[string]float32)
	for i, divisor := range r.divisors {
		divisors[strconv.Itoa(i)] = divisor
	}
	jsonBytes, e := json.Marshal(divisors)
	if e != nil {
		r.errorHandler.Error(e)
		return e
	}

	e = ioutil.WriteFile(configFile, jsonBytes, os.ModePerm)
	if e != nil {
		r.errorHandler.Error(e)
		return e
	}

	return nil
}

func (r *RegressionDivisor) Fit(input []float32) {
	for i := 0; i < len(input); i++ {
		if len(r.divisors) <= i {
			if input[i] == 0 {
				r.divisors = append(r.divisors, 1)
			} else {
				r.divisors = append(r.divisors, input[i])
			}
		} else if input[i] > r.divisors[i] {
			r.divisors[i] = input[i]
		}
	}
}

func (r *RegressionDivisor) Divide(input []float32) ([]float32, error) {
	var divided []float32

	for offset, value := range input {
		if len(r.divisors) <= offset {
			return nil, fmt.Errorf("mising divisor %d, divisors len: %d", offset, len(r.divisors))
		}
		divided = append(divided, value/r.divisors[offset])
	}

	return divided, nil
}

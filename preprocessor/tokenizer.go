package preprocessor

import (
	"encoding/json"
	"github.com/codingbeard/cberrors"
	"io/ioutil"
	"os"
	"sort"
	"strings"
	"sync"
)

type Tokenizer struct {
	isCategoryTokenizer bool
	dictionary          map[string]int
	wordCounts          map[string]int
	maxLen              int
	numWords            int
	lock                *sync.Mutex

	errorHandler *cberrors.ErrorsContainer
}

type tokenizerConfig struct {
	MaxLen    int            `json:"max_len"`
	WordIndex map[string]int `json:"word_index"`
}

func NewTokenizer(
	errorHandler *cberrors.ErrorsContainer,
	maxLen int,
	numWords int,
	isCategoryTokenizer bool,
) *Tokenizer {
	if isCategoryTokenizer {
		numWords = 1000000
	}
	return &Tokenizer{
		isCategoryTokenizer: isCategoryTokenizer,
		dictionary:          make(map[string]int),
		wordCounts:          make(map[string]int),
		maxLen:              maxLen,
		numWords:            numWords,
		lock:                &sync.Mutex{},
		errorHandler:        errorHandler,
	}
}

func (t *Tokenizer) Load(configFile string) error {
	contents, e := ioutil.ReadFile(configFile)
	if e != nil {
		return e
	}

	var config tokenizerConfig

	e = json.Unmarshal(contents, &config)
	if e != nil {
		return e
	}

	t.dictionary = config.WordIndex
	t.maxLen = config.MaxLen

	return nil
}

func (t *Tokenizer) Save(configFile string) error {
	jsonBytes, e := json.Marshal(tokenizerConfig{
		MaxLen:    t.maxLen,
		WordIndex: t.dictionary,
	})
	if e != nil {
		t.errorHandler.Error(e)
		return e
	}

	e = ioutil.WriteFile(configFile, jsonBytes, os.ModePerm)
	if e != nil {
		t.errorHandler.Error(e)
		return e
	}

	return nil
}

func (t *Tokenizer) clean(sentence string) string {
	for strings.Contains(sentence, "  ") {
		sentence = strings.ReplaceAll(sentence, "  ", " ")
	}

	sentence = strings.ReplaceAll(sentence, "\x00", "")

	sentence = strings.ToLower(sentence)

	return sentence
}

func (t *Tokenizer) split(sentence string) []string {
	return strings.Split(strings.TrimSpace(sentence), " ")
}

func (t *Tokenizer) Fit(sentence string) {
	words := t.split(t.clean(sentence))

	for _, word := range words {
		t.lock.Lock()
		count := t.wordCounts[word]
		count++
		t.wordCounts[word] = count
		t.lock.Unlock()
	}

	if len(t.wordCounts) > t.numWords*10 {
		type kv struct {
			k string
			v int
		}
		var kvs []kv
		t.lock.Lock()
		for word, count := range t.wordCounts {
			kvs = append(kvs, kv{
				k: word,
				v: count,
			})
		}

		sort.Slice(kvs, func(i, j int) bool {
			return kvs[i].v > kvs[j].v
		})

		for i := t.numWords; i < len(kvs); i++ {
			delete(t.wordCounts, kvs[i].k)
		}

		t.lock.Unlock()
	}
}

func (t *Tokenizer) FinishFit() {
	t.lock.Lock()
	defer t.lock.Unlock()

	type kv struct {
		k string
		v int
	}
	var kvs []kv

	for word, count := range t.wordCounts {
		kvs = append(kvs, kv{
			k: word,
			v: count,
		})
	}

	sort.Slice(kvs, func(i, j int) bool {
		return kvs[i].v > kvs[j].v
	})

	totalWords := len(kvs)
	if t.numWords < totalWords {
		totalWords = t.numWords
	}

	delta := 1
	if t.isCategoryTokenizer {
		delta = 0
	}

	for i := 0; i < totalWords; i++ {
		t.dictionary[kvs[i].k] = i + delta
	}
}

func (t *Tokenizer) Tokenize(sentence string) []int32 {
	tokenized := make([]int32, t.maxLen)

	words := t.split(t.clean(sentence))

	position := 0
	for _, word := range words {
		if position >= t.maxLen {
			break
		}
		word = strings.TrimSpace(strings.ToLower(word))

		t.lock.Lock()
		dictionaryIndex, ok := t.dictionary[word]
		t.lock.Unlock()

		if ok {
			tokenized[position] = int32(dictionaryIndex)
			position++
		}
	}

	return tokenized
}

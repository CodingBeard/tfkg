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
	wordCounts          map[string]*wordCount
	uniqueWordCount     int
	maxLen              int
	numWords            int
	filter              string
	disableFiltering    bool
	lock                *sync.Mutex

	errorHandler *cberrors.ErrorsContainer
}

type wordCount struct {
	count int
	order int
}

type tokenizerConfig struct {
	MaxLen           int            `json:"max_len"`
	WordIndex        map[string]int `json:"word_index"`
	Filter           string         `json:"filter"`
	DisableFiltering bool           `json:"disable_filtering"`
}

type TokenizerConfig struct {
	IsCategoryTokenizer bool
	Filters             string
	DisableFiltering    bool
}

func NewTokenizer(
	errorHandler *cberrors.ErrorsContainer,
	maxLen int,
	numWords int,
	configs ...TokenizerConfig,
) *Tokenizer {
	config := TokenizerConfig{}
	if len(configs) > 0 {
		config = configs[0]
	}
	if config.IsCategoryTokenizer {
		numWords = 1000000
	}
	if config.Filters == "" {
		config.Filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n"
	}
	return &Tokenizer{
		isCategoryTokenizer: config.IsCategoryTokenizer,
		dictionary:          make(map[string]int),
		wordCounts:          make(map[string]*wordCount),
		maxLen:              maxLen,
		numWords:            numWords,
		filter:              config.Filters,
		disableFiltering:    config.DisableFiltering,
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
	t.numWords = len(config.WordIndex)
	t.filter = config.Filter
	t.disableFiltering = config.DisableFiltering

	return nil
}

func (t *Tokenizer) Save(configFile string) error {
	jsonBytes, e := json.Marshal(tokenizerConfig{
		MaxLen:           t.maxLen,
		WordIndex:        t.dictionary,
		Filter:           t.filter,
		DisableFiltering: t.disableFiltering,
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

func (t *Tokenizer) NumWords() int {
	return len(t.dictionary)
}

func (t *Tokenizer) MaxLen() int {
	return t.maxLen
}

func (t *Tokenizer) clean(sentence string) string {
	sentence = strings.ReplaceAll(sentence, "\x00", "")
	for strings.Contains(sentence, "  ") {
		sentence = strings.ReplaceAll(sentence, "  ", " ")
	}

	if !t.disableFiltering {
		for _, char := range t.filter {
			sentence = strings.ReplaceAll(sentence, string(char), "")
		}
	}

	sentence = strings.ToLower(sentence)

	return sentence
}

func (t *Tokenizer) split(sentence string) []string {
	return strings.Split(strings.TrimSpace(sentence), " ")
}

func (t *Tokenizer) Fit(sentence string) {
	previousUniqueWordCount := t.uniqueWordCount
	words := t.split(t.clean(sentence))

	for _, word := range words {
		t.lock.Lock()
		count := t.wordCounts[word]
		if count == nil {
			count = &wordCount{
				count: 1,
				order: t.uniqueWordCount,
			}
			t.uniqueWordCount++
		} else {
			count.count++
		}
		t.wordCounts[word] = count
		t.lock.Unlock()
	}

	if len(t.wordCounts) > t.numWords*200 {
		type kv struct {
			k string
			v int
		}
		var kvs []kv
		t.lock.Lock()
		for word, count := range t.wordCounts {
			kvs = append(kvs, kv{
				k: word,
				v: count.count,
			})
		}

		sort.Slice(kvs, func(i, j int) bool {
			return kvs[i].v > kvs[j].v
		})

		for i := t.numWords * 100; i < len(kvs); i++ {
			delete(t.wordCounts, kvs[i].k)
		}

		t.lock.Unlock()
	}
	if t.isCategoryTokenizer {
		if t.uniqueWordCount > previousUniqueWordCount {
			t.FinishFit()
		}
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

	if t.isCategoryTokenizer {
		t.dictionary = make(map[string]int)
		for word, count := range t.wordCounts {
			kvs = append(kvs, kv{
				k: word,
				v: count.order,
			})
		}

		sort.Slice(kvs, func(i, j int) bool {
			return kvs[i].v < kvs[j].v
		})

		for i, count := range kvs {
			t.dictionary[count.k] = i
		}
		return
	}

	for word, count := range t.wordCounts {
		kvs = append(kvs, kv{
			k: word,
			v: count.count,
		})
	}

	sort.Slice(kvs, func(i, j int) bool {
		return kvs[i].v > kvs[j].v
	})

	totalWords := len(kvs)
	if t.numWords < totalWords {
		totalWords = t.numWords
	}

	for i := 0; i < totalWords; i++ {
		t.dictionary[kvs[i].k] = i + 1
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

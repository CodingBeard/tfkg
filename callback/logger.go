package callback

import (
	"errors"
	"fmt"
	"github.com/codingbeard/cblog"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	LoggerVerbose      = "verbose"
	LoggerTotalBatches = "totalBatches"
	LoggerPrefetched   = "prefetched"
)

type Logger struct {
	FileLogger     *cblog.Logger
	Progress       bool
	ProgressLogDir string
	modeStarts     map[Mode]int64
	modeStartsLock *sync.Mutex
	lastPrint      int64
}

func (l *Logger) Init() error {
	if l.FileLogger == nil {
		return errors.New("no FileLogger set on logger callback")
	}
	if l.Progress && l.ProgressLogDir == "" {
		return errors.New("progress is enabled but no ProgressLogDir set on logger callback")
	}
	l.modeStartsLock = &sync.Mutex{}
	l.modeStarts = make(map[Mode]int64)
	return nil
}

func (l *Logger) Call(event Event, mode Mode, epoch int, batch int, logs []Log) ([]Action, error) {
	l.modeStartsLock.Lock()
	defer l.modeStartsLock.Unlock()
	if event == EventStart {
		if mode == ModeVal {
			if l.modeStarts[ModeTrain] == 0 {
				l.modeStarts[ModeTrain] = time.Now().Unix()
			}
		} else if mode == ModeTest {
			if l.modeStarts[ModeTrain] == 0 {
				l.modeStarts[ModeTrain] = time.Now().Unix()
			}
			if l.modeStarts[ModeVal] == 0 {
				l.modeStarts[ModeVal] = time.Now().Unix()
			}
		}
		l.modeStarts[mode] = time.Now().Unix()
		l.lastPrint = 0
		return []Action{ActionNop}, nil
	}

	totalBatches := 0
	var metricString []string
	var metricValues []interface{}
	prefetched := -1
	verbose := false

	for _, log := range logs {
		if log.Name == LoggerTotalBatches {
			totalBatches = int(log.Value)
		} else if log.Name == LoggerPrefetched {
			prefetched = int(log.Value)
		} else if log.Name == LoggerVerbose {
			if log.Value != -1 {
				verbose = true
			}
		} else {
			metricString = append(metricString, log.Name+": %."+strconv.Itoa(log.Precision)+"f")
			metricValues = append(metricValues, log.Value)
		}
	}

	if event == EventSave {
		if verbose {
			l.FileLogger.InfoF("training", "Saved")
		}
		return []Action{ActionNop}, nil
	}

	previousElapsed := int64(0)
	elapsedInMode := time.Now().Unix() - l.modeStarts[mode]

	if mode == ModeVal {
		previousElapsed = time.Now().Unix() - l.modeStarts[ModeTrain] - elapsedInMode
	} else if mode == ModeTest {
		previousElapsed = time.Now().Unix() - l.modeStarts[ModeTrain] + time.Now().Unix() - l.modeStarts[ModeVal] - elapsedInMode
	}

	logType := strings.Title(string(mode))
	if event == EventEnd {
		logType = "End"
	}

	if event == EventEnd && (mode == ModeVal || mode == ModeTest) {
		if verbose {
			fmt.Print("\r")
		}
		l.FileLogger.InfoF("training", fmt.Sprintf(
			"%s %d %d/%d (%ds/%ds) %s                 ",
			logType,
			epoch,
			batch,
			totalBatches,
			previousElapsed+elapsedInMode,
			previousElapsed+int64((float32(elapsedInMode)/float32(batch))*float32(totalBatches)),
			fmt.Sprintf(
				strings.Join(metricString, " "),
				metricValues...,
			),
		))
	} else if verbose {
		now := time.Now().Unix()
		if now > l.lastPrint {
			l.lastPrint = now
			log := fmt.Sprintf(
				"\r%s : logger.go : %s %d %d/%d (%ds/%ds) %s | Prefetched %d",
				time.Now().Format("2006-01-02 15:04:05.000"),
				logType,
				epoch,
				batch,
				totalBatches,
				previousElapsed+elapsedInMode,
				previousElapsed+int64((float32(elapsedInMode)/float32(batch))*float32(totalBatches)),
				fmt.Sprintf(
					strings.Join(metricString, " "),
					metricValues...,
				),
				prefetched,
			)
			fmt.Print(log)
			if l.Progress {
				e := ioutil.WriteFile(filepath.Join(l.ProgressLogDir, "progress.log"), []byte(log[1:]), os.ModePerm)
				if e != nil {
					l.FileLogger.ErrorF("error", e.Error())
				}
			}
		}
	}

	return []Action{ActionNop}, nil
}

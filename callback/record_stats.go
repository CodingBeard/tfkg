package callback

import (
	"encoding/csv"
	"github.com/codingbeard/cbutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type RecordStats struct {
	RecordDir      string
	RecordFileName string
	OnEvent        Event
	OnMode         Mode

	recordFile   *os.File
	recordWriter *csv.Writer
}

func (r *RecordStats) Init() error {
	file, e := os.Create(filepath.Join(r.RecordDir, r.RecordFileName))
	if e != nil {
		return e
	}
	writer := csv.NewWriter(file)

	r.recordFile = file
	r.recordWriter = writer
	return nil
}

func (r *RecordStats) Call(event Event, mode Mode, epoch int, batch int, logs []Log) ([]Action, error) {
	if event == r.OnEvent && mode == r.OnMode {
		if epoch == 1 {
			columns := []string{
				"datetime",
				"event",
				"mode",
				"epoch",
				"batch",
			}

			for _, log := range logs {
				columns = append(columns, strings.ToLower(log.Name))
			}

			e := r.recordWriter.Write(columns)
			if e != nil {
				return []Action{ActionNop}, e
			}
			r.recordWriter.Flush()
		}

		values := []string{
			time.Now().Format(cbutil.DateTimeFormat),
			string(event),
			string(mode),
			strconv.Itoa(epoch),
			strconv.Itoa(batch),
		}

		for _, log := range logs {
			values = append(values, strconv.FormatFloat(float64(log.Value), 'f', -1, 32))
		}

		e := r.recordWriter.Write(values)
		if e != nil {
			return nil, e
		}
		r.recordWriter.Flush()
	}

	return []Action{ActionNop}, nil
}

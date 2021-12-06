package callback

type Mode string
type Event string
type Action string

var (
	ModeTrain   Mode   = "train"
	ModeVal     Mode   = "val"
	ModeTest    Mode   = "test"
	EventStart  Event  = "start"
	EventDuring Event  = "during"
	EventEnd    Event  = "end"
	EventSave   Event  = "save"
	ActionNop   Action = "nop"
	ActionSave  Action = "save"
	ActionHalt  Action = "halt"
)

type Log struct {
	Name      string
	Value     float64
	Precision int
}

type Callback interface {
	Init() error
	Call(event Event, mode Mode, epoch int, batch int, logs []Log) ([]Action, error)
}

type HasSaveDir interface {
	GetSaveDir() string
}

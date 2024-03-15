package seafan

// ch.go implements a Pipeline using github.com/invertedv/chutils

import (
	"fmt"
	"io"
	"reflect"

	"github.com/invertedv/chutils"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ChData provides a Pipeline interface into text files (delimited, fixed length) and ClickHouse.
type ChData struct {
	cycle      bool          // if true, reuses data, false fetches new data after each epoch
	pull       bool          // if true, pull the data from ClickHouse on next call to Batch
	bs         int           // batch size
	cbRow      int           // current batch starting row
	nRow       int           // # rows in dataset
	rdr        chutils.Input // data reader
	data       *GData        // processed data
	epochCount int           // current epoch
	ftypes     FTypes        // user input selections
	keepRaw    bool
	callback   Opts   // user callbacks executed at the start of Init()
	name       string // pipeline name
}

func NewChData(name string, opts ...Opts) *ChData {
	ch := &ChData{bs: 1, cycle: true, pull: true, name: name}
	for _, o := range opts {
		o(ch)
	}

	return ch
}

// GetKeepRaw returns true if *Raw data is retained
func (ch *ChData) GetKeepRaw() bool {
	return ch.keepRaw
}

// GetFTypes returns FTypes for ch Pipeline.
func (ch *ChData) GetFTypes() FTypes {
	if ch.data == nil {
		return nil
	}

	return ch.data.GetFTypes()
}

// SaveFTypes saves the FTypes for the Pipeline.
func (ch *ChData) SaveFTypes(fileName string) error {
	return ch.GetFTypes().Save(fileName)
}

// returns *FType from user-input FTypes.
func (ch *ChData) getFType(feature string) *FType {
	if ch.ftypes == nil {
		return nil
	}

	return ch.ftypes.Get(feature)
}

// IsNormalized returns true if the field is normalized.
func (ch *ChData) IsNormalized(field string) bool {
	if ft := ch.Get(field); ft != nil {
		return ft.FT.Normalized
	}

	return false
}

// IsCts returns true if the field has role FRCts.
func (ch *ChData) IsCts(field string) bool {
	if ft := ch.Get(field); ft != nil {
		return ft.FT.Role == FRCat
	}

	return false
}

// IsCat returns true if field has role FRCat.
func (ch *ChData) IsCat(field string) bool {
	if ft := ch.Get(field); ft != nil {
		return ft.FT.Role == FRCat
	}

	return false
}

// GData returns the Pipelines' GData
func (ch *ChData) GData() *GData {
	d := ch.data

	return d
}

// Init initializes the Pipeline.
func (ch *ChData) Init() (err error) {
	if ch.rdr == nil {
		return Wrapper(ErrChData, "no reader")
	}

	nRow, e := ch.rdr.CountLines()
	if e != nil {
		panic(e)
	}

	ch.nRow = nRow
	if ch.bs == 0 {
		ch.bs = nRow
	}

	if ch.bs > ch.nRow {
		return Wrapper(ErrChData, fmt.Sprintf("Init: batch size = %d > dataset rows = %d", ch.bs, ch.nRow))
	}

	ch.pull = false
	fds := ch.rdr.TableSpec().FieldDefs
	names := make([]string, len(fds))           // field names
	trans := make([]*Raw, len(fds))             // data
	chTypes := make([]chutils.ChType, len(fds)) // field types

	for ind := 0; ind < len(fds); ind++ {
		names[ind] = fds[ind].Name
		chTypes[ind] = fds[ind].ChSpec.Base
	}

	// load GData
	anyData := false
	for rw := 0; ; rw++ {
		r, _, ex := ch.rdr.Read(1, true)
		if ex != nil && ex != io.EOF {
			return ex
		}

		if ex == io.EOF {
			if Verbose {
				fmt.Println("rows read: ", rw)
			}

			break
		}

		// now we have the types, we can allocate the slices
		if rw == 0 {
			for c := 0; c < len(r[0]); c++ {
				trans[c] = AllocRaw(nRow, reflect.TypeOf(r[0][c]).Kind())
			}
		}

		anyData = true
		for c := 0; c < len(trans); c++ {
			trans[c].Data[rw] = r[0][c]
		}
	}

	if !anyData {
		return fmt.Errorf("ch.Init failed...query EOF with no data")
	}

	gd := NewGData()

	// work through fields, add to GData
	for ind, nm := range names {
		// if this isn't in our array, add it
		ft := ch.getFType(nm) // note: this version gets user-Inputs
		if ft == nil {
			ft = &FType{}

			switch chTypes[ind] {
			case chutils.ChDate, chutils.ChString, chutils.ChFixedString:
				ft.Role = FRCat
			default:
				ft.Role = FRCts
			}
		}

		switch ft.Role {
		case FRCts:
			if err = gd.AppendC(trans[ind], nm, ft.Normalized, ft.FP, ch.keepRaw); err != nil {
				return Wrapper(err, "(*ChData).Init")
			}
		default:
			if err = gd.AppendD(trans[ind], names[ind], ft.FP, ch.keepRaw); err != nil {
				return Wrapper(err, "(*ChData).Init")
			}
		}
	}
	// Add calculated fields
	for _, ft := range ch.ftypes {
		switch ft.Role {
		case FROneHot:
			if err = gd.MakeOneHot(ft.From, ft.Name); err != nil {
				return Wrapper(err, "(*ChData).Init")
			}
		case FREmbed:
			if err = gd.MakeOneHot(ft.From, ft.Name); err != nil {
				return Wrapper(err, "(*ChData).Init")
			}
		}
	}

	ch.data = gd

	return nil
}

// Rows is # of rows of data in the Pipeline
func (ch *ChData) Rows() int {
	return ch.nRow
}

// Batch loads a batch into Inputs.  It returns false if the epoch is done.
// If cycle is true, it will start at the beginning on the next call.
// If cycle is false, it will call Init() at the next call to Batch()
func (ch *ChData) Batch(inputs G.Nodes) bool {
	// do we need to load the data?
	if ch.pull {
		if e := ch.rdr.Reset(); e != nil {
			panic(e)
		}

		if e := ch.Init(); e != nil {
			panic(e)
		}
	}
	// out of data?  if NRows % bsize !=0, rows after the last full batch are unused.
	if ch.cbRow+ch.bs > ch.nRow {
		if !ch.cycle {
			ch.pull = true
		}

		ch.cbRow = 0
		// user callbacks
		if ch.callback != nil {
			ch.callback(ch)
		}

		return false
	}

	startRow := ch.cbRow
	endRow := startRow + ch.bs

	for _, nd := range inputs {
		var t tensor.Tensor

		d := ch.data.Get(nd.Name())

		if d == nil {
			panic(Wrapper(ErrChData, fmt.Sprintf("feature %s not in dataset", nd.Name())))
		}

		switch d.FT.Role {
		case FRCts:
			t = tensor.New(tensor.WithBacking(d.Data.([]float64)[startRow:endRow]), tensor.WithShape(ch.bs, 1))
		case FRCat:
			t = tensor.New(tensor.WithBacking(d.Data.([]int32)[startRow:endRow]), tensor.WithShape(ch.bs, 1))
		case FROneHot, FREmbed:
			sr := startRow * d.FT.Cats
			er := endRow * d.FT.Cats
			t = tensor.New(tensor.WithBacking(d.Data.([]float64)[sr:er]), tensor.WithShape(ch.bs, d.FT.Cats))
		}

		if e := G.Let(nd, t); e != nil {
			panic(e)
		}
	}

	ch.cbRow = endRow

	return true
}

// Get returns a fields's GDatum
func (ch *ChData) Get(field string) *GDatum {
	return ch.data.Get(field)
}

// Cols returns the # of columns in the field
func (ch *ChData) Cols(field string) int {
	d := ch.Get(field)

	if d == nil {
		return 0
	}

	switch d.FT.Role {
	case FRCts:
		return 1
	case FRCat:
		return 1
	case FROneHot, FREmbed:
		return d.FT.Cats
	}

	return 0
}

// Epoch sets the epoch to setTo if setTo >=0.
// Returns epoch #.
func (ch *ChData) Epoch(setTo int) int {
	if setTo >= 0 {
		ch.epochCount = setTo
	}

	return ch.epochCount
}

// FieldList returns a slice of field names in the Pipeline
func (ch *ChData) FieldList() []string {
	fl := make([]string, 0)
	for _, ft := range ch.data.data {
		fl = append(fl, ft.FT.Name)
	}

	return fl
}

// GetFType returns the field's FType
func (ch *ChData) GetFType(field string) *FType {
	if ch.data == nil {
		return nil
	}

	return ch.data.GetFType(field)
}

// Name returns Pipeline name
func (ch *ChData) Name() string {
	return ch.name
}

// BatchSize returns Pipeline batch size.  Use WithBatchSize to set this.
func (ch *ChData) BatchSize() int {
	return ch.bs
}

// Describe describes a field.  If the field has role FRCat, the top k values (by frequency) are returned.
func (ch *ChData) Describe(field string, topK int) string {
	d := ch.Get(field)
	if d == nil {
		return ""
	}

	return d.Describe(topK)
}

func (ch *ChData) String() string {
	const numCats = 5
	str := fmt.Sprintf("Summary for pipeline %s\n", ch.Name())
	fl := ch.FieldList()
	str = fmt.Sprintf("%s%d fields\n", str, len(fl))

	for _, f := range fl {
		ff := ch.Describe(f, numCats)
		str += "\n" + ff
	}

	return str
}

// Slice returns a VecData Pipeline sliced according to sl
func (ch *ChData) Slice(sl Slicer) (Pipeline, error) {
	gData, e := ch.data.Slice(sl)
	if e != nil {
		return nil, Wrapper(e, "*ChData.Slice")
	}

	name := fmt.Sprintf("sliced from %s", ch.name)
	vecData := NewVecData(name, gData)

	return vecData, nil
}

// Shuffle shuffles the data
func (ch *ChData) Shuffle() {
	ch.data.Shuffle()
}

// Sort sorts the data
func (ch *ChData) Sort(field string, ascending bool) error {
	e := ch.data.Sort(field, ascending)
	if e != nil {
		return Wrapper(e, "(*ChData) Sort")
	}
	return nil
}

// IsSorted returns true if the data has been sorted.
func (ch *ChData) IsSorted() bool {
	return ch.data.IsSorted()
}

// SortField returns the field the data is sorted on.
func (ch *ChData) SortField() string {
	return ch.data.SortField()
}

// Row creates a new pipeline with only the row, take
func (ch *ChData) Row(take int) (newPipe Pipeline, err error) {
	var gdNew *GData

	if gdNew, err = ch.GData().Row(take); err != nil {
		return nil, err
	}

	newPipe = NewVecData("new pipe", gdNew)
	WithKeepRaw(ch.keepRaw)(newPipe)

	return newPipe, nil
}

// Subset creates a new pipeline with only the rows, rows
func (ch *ChData) Subset(rows []int) (newPipe Pipeline, err error) {
	var gdNew *GData

	if gdNew, err = ch.GData().Subset(rows); err != nil {
		return nil, err
	}

	newPipe = NewVecData("new pipe", gdNew)
	WithKeepRaw(ch.keepRaw)(newPipe)

	return newPipe, nil
}

// Where creates a new pipeline with rows where field is in equalTo. The comparison uses the *Raw data.
func (ch *ChData) Where(field string, equalTo []any) (newPipe Pipeline, err error) {
	var gdNew *GData

	if gdNew, err = ch.GData().Where(field, equalTo); err != nil {
		return nil, err
	}

	newPipe = NewVecData("new pipe", gdNew)
	WithKeepRaw(ch.keepRaw)(newPipe)

	return newPipe, nil
}

// FieldCount returns the number of fields in the pipeline
func (ch *ChData) FieldCount() int {
	return ch.data.FieldCount()
}

// Keep keeps only the listed fields in the pipeline
func (ch *ChData) Keep(fields []string) error {
	return ch.GData().Keep(fields)
}

// Drop drops the listed field from the pipeline
func (ch *ChData) Drop(field string) error {
	return ch.GData().Drop(field)
}

// AppendRows appends rows to the existing GData and then re-initializes each GDatum, using the fTypes, if provided.
func (ch *ChData) AppendRows(gd *GData, fTypes FTypes) (pipeOut Pipeline, err error) {
	gdOut, e := ch.GData().AppendRows(gd, fTypes)
	if e != nil {
		return nil, e
	}

	pipeOut = NewVecData("out", gdOut)
	WithKeepRaw(ch.keepRaw)(pipeOut)

	return pipeOut, nil
}

// AppendRowsRaw simply appends rows, in place, to the existing GData.  Only the *Raw data is updated.
// The .Data field is set to nil.
func (ch *ChData) AppendRowsRaw(gd *GData) error {
	ch.nRow += gd.rows

	return ch.GData().AppendRowsRaw(gd)
}

// ReInit re-initializes the Data field from Raw for each GDatum. If ftypes is not nil, these values
// are used, otherwise the FParam values are re-derived from the data. A new pipeline is returned.
func (ch *ChData) ReInit(ftypes *FTypes) (pipeOut Pipeline, err error) {
	var gdNew *GData

	if gdNew, err = ch.GData().ReInit(ftypes); err != nil {
		return nil, err
	}

	pipeOut = NewVecData("new", gdNew)
	WithKeepRaw(ch.keepRaw)(pipeOut)

	return pipeOut, nil
}

func (ch *ChData) Join(right Pipeline, onField string, joinType JoinType) (result Pipeline, err error) {
	gdResult, e := ch.data.Join(right.GData(), onField, joinType)
	if e != nil {
		return nil, e
	}

	result = NewVecData("joined", gdResult)
	WithKeepRaw(ch.keepRaw)(result)

	return result, nil
}

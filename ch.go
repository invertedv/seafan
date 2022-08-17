package seafan

// github.com/invertedv/chutils Pipeline

import (
	"fmt"
	"github.com/invertedv/chutils"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"io"
	"log"
	"reflect"
	"strings"
)

// ChData creates a Pipeline based on chutils.  This provides an interface into text files (delimited, fixed length)
// and ClickHouse.
// If !cycle, Init is called after each epoch.
// callback is called at the start of each Init
type ChData struct {
	bs         int           // batch size
	cycle      bool          // if true, resuses data, false fetches new data after each epoch
	cbRow      int           // current batch starting row
	pull       bool          // if true, pull the data from ClickHouse on next call to Batch
	nRow       int           // # rows in dataset
	rdr        chutils.Input // data reader
	data       GData         // processed data
	epochCount int           // current epoch
	ftypes     FTypes        // user input selections
	callback   Opts          // user callbacks executed at the start of Init()
	name       string        // pipeline name
}

func NewChData(name string, opts ...Opts) *ChData {
	ch := &ChData{bs: 1, cycle: true, pull: true, name: name}
	for _, o := range opts {
		o(ch)
	}
	return ch
}

// GetFtypes returns FTypes for ch Pipeline
func (ch *ChData) GetFTypes() FTypes {
	fts := make(FTypes, 0)
	for _, d := range ch.data {
		fts = append(fts, d.FT)
	}
	return fts
}

// SaveFTypes saves the FTypes for the Pipeline.
func (ch *ChData) SaveFTypes(fileName string) error {
	return ch.GetFTypes().Save(fileName)
}

// returns *FType from user-input FTypes
func (ch *ChData) getFType(feature string) *FType {
	for _, ft := range ch.ftypes {
		if ft.Name == feature {
			return ft
		}
	}
	return nil
}

// IsNormalized returns true if the field is normalized
func (ch *ChData) IsNormalized(field string) bool {
	if ft := ch.Get(field); ft != nil {
		return ft.FT.Normalized
	}
	return false
}

// IsCts returns true if the field has role FRCts
func (ch *ChData) IsCts(field string) bool {
	if ft := ch.Get(field); ft != nil {
		return ft.FT.Role == FRCat
	}
	return false

}

// IsCat returns true if field has role FRCat
func (ch *ChData) IsCat(field string) bool {
	if ft := ch.Get(field); ft != nil {
		return ft.FT.Role == FRCat
	}
	return false
}

// GData returns the Pipelines' GData
func (ch *ChData) GData() GData {
	d := ch.data
	return d
}

// Init initializes the Pipeline.
func (ch *ChData) Init() (err error) {
	err = nil
	// user callbacks
	if ch.callback != nil {
		ch.callback(ch)
	}
	if ch.rdr == nil {
		return fmt.Errorf("no reader")
	}
	nRow, e := ch.rdr.CountLines()
	if e != nil {
		log.Fatalln(e)
	}
	ch.nRow = nRow
	if ch.bs > ch.nRow {
		return fmt.Errorf("batch size = %d > dataset rows = %d", ch.bs, ch.nRow)
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
	for row := 0; ; row++ {
		r, _, e := ch.rdr.Read(1, true)
		if e == io.EOF {
			fmt.Println("rows read: ", row)
			break
		}
		// now we have the types, we can allocate the slices
		if row == 0 {
			for c := 0; c < len(r[0]); c++ {
				trans[c] = AllocRaw(nRow, reflect.TypeOf(r[0][c]).Kind())
			}
		}
		for c := 0; c < len(trans); c++ {
			trans[c].Data[row] = r[0][c]
		}
	}
	gd := make(GData, 0)

	// work through fields, add to GData
	for ind, nm := range names {
		// if this isn't in our array, add it
		ft := ch.getFType(nm)
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
			if gd, err = gd.AppendC(trans[ind], nm, ft.Normalized, ft.FP); err != nil {
				return
			}
		default:
			if gd, err = gd.AppendD(trans[ind], names[ind], ft.FP); err != nil {
				return
			}
		}
	}

	// Add calculated fields
	for _, ft := range ch.ftypes {
		switch ft.Role {
		case FROneHot:
			if gd, err = gd.MakeOneHot(ft.From, ft.Name); err != nil {
				return
			}
		case FREmbed:
			if gd, err = gd.MakeOneHot(ft.From, ft.Name); err != nil {
				return
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

// Batch loads a batch into inputs.  It returns false if the epoch is done.
// If cycle, it will start at the beginning on the next call.
// If !cycle, it will call Init() at the next call to Batch()
func (ch *ChData) Batch(inputs G.Nodes) bool {
	// do we need to load the data?
	if ch.pull {
		if e := ch.rdr.Reset(); e != nil {
			log.Fatalln(e)
		}
		if e := ch.Init(); e != nil {
			log.Fatalln(e)
		}
	}
	// out of Data?  if nRow % bsize !=0, rows after the last full batch are unused.
	if ch.cbRow+ch.bs > ch.nRow {
		if !ch.cycle {
			ch.pull = true
		}
		ch.cbRow = 0
		return false
	}
	startRow := ch.cbRow
	endRow := startRow + ch.bs
	for _, nd := range inputs {
		var t tensor.Tensor
		d := ch.data.Get(nd.Name())
		if d == nil {
			log.Fatalln(fmt.Errorf("feature %s not in dataset", nd.Name()))
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
			log.Fatalln(e)
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
	for _, ft := range ch.data {
		fl = append(fl, ft.FT.Name)
	}
	return fl
}

// GetFType returns the fields FType
func (ch *ChData) GetFType(field string) *FType {
	d := ch.Get(field)
	if d == nil {
		return nil
	}
	return d.FT
}

// Name returns Pipeline name
func (ch *ChData) Name() string {
	return ch.name
}

// BatchSize returns Pieeline batch size
func (ch *ChData) BatchSize() int {
	return ch.bs
}

// Describe describes a field.  If the field has role FRCat, the top k values (by frequency) are returned.
func (ch *ChData) Describe(field string, topK int) string {
	d := ch.Get(field)
	if d == nil {
		return ""
	}
	if topK <= 0 {
		topK = 5
	}
	topK = Max(Min(topK, 100), 2)

	str := d.FT.String()
	switch d.FT.Role {
	case FRCts:
		str = fmt.Sprintf("%s%s", str, "\t"+strings.ReplaceAll(d.Summary.DistrC.String(), "\n", "\n\t"))
	case FRCat:
		str = fmt.Sprintf("%s\tTop 5 Values\n", str)
		str = fmt.Sprintf("%s%s", str, "\t"+strings.ReplaceAll(d.Summary.DistrD.TopK(topK, false, false), "\n", "\n\t"))
	}
	return str
}

func (ch *ChData) String() string {
	str := fmt.Sprintf("Summary for pipeline %s\n", ch.Name())
	fl := ch.FieldList()
	str = fmt.Sprintf("%s%d fields\n", str, len(fl))
	for _, f := range fl {
		ff := ch.Describe(f, 5)
		str += ff
	}
	return str
}

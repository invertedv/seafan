package seafan

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

// ChData creates a Pipeline based on ClickHouse access using chutils
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
	callback   Opts          // user callbacks at end of each epoch
	name       string        // pipeline name
}

func NewChData(name string, opts ...Opts) *ChData {
	ch := &ChData{bs: 1, cycle: true, pull: true, name: name}
	for _, o := range opts {
		o(ch)
	}
	return ch
}

func (ch *ChData) SaveFTypes(fileName string) error {
	fts := make(FTypes, 0)
	for _, d := range ch.data {
		fts = append(fts, d.FT)
	}
	return fts.Save(fileName)
}

func (ch *ChData) getFTypes(feature string) *FType {
	for _, ft := range ch.ftypes {
		if ft.Name == feature {
			return ft
		}
	}
	return nil
}

func (ch *ChData) IsNormalized(name string) bool {
	if ft := ch.Get(name); ft != nil {
		return ft.FT.Normalized
	}
	return false
}

func (ch *ChData) IsCts(name string) bool {
	return !ch.IsCat(name)
}

func (ch *ChData) IsCat(name string) bool {
	if ft := ch.Get(name); ft != nil {
		return ft.FT.Role == FRCat
	}
	return false
}

func (ch *ChData) GData() GData {
	d := ch.data
	return d
}

func (ch *ChData) Init() (err error) {
	err = nil
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
	names := make([]string, len(fds))
	trans := make([]*Raw, len(fds))
	chTypes := make([]chutils.ChType, len(fds))
	for ind := 0; ind < len(fds); ind++ {
		names[ind] = fds[ind].Name
		chTypes[ind] = fds[ind].ChSpec.Base
	}
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

	for ind, nm := range names {
		// if this isn't in our array, add it
		ft := ch.getFTypes(nm)
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

func (ch *ChData) Rows() int {
	return ch.nRow
}

func (ch *ChData) Batch(inputs G.Nodes) bool {
	if ch.pull {
		if e := ch.rdr.Reset(); e != nil {
			log.Fatalln(e)
		}
		if e := ch.Init(); e != nil {
			log.Fatalln(e)
		}
	}
	// out of data?  if nRow % bsize !=0, rows after the last full batch are unused.
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

// Get returns a feature
func (ch *ChData) Get(name string) *GDatum {
	return ch.data.Get(name)
}

func (ch *ChData) Cols(feature string) int {
	d := ch.Get(feature)
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

func (ch *ChData) Epoch(setTo int) int {
	if setTo >= 0 {
		ch.epochCount = setTo
	}
	return ch.epochCount

}

func (ch *ChData) FieldList() []string {
	fl := make([]string, 0)
	for _, ft := range ch.data {
		fl = append(fl, ft.FT.Name)
	}
	return fl
}

func (ch *ChData) GetFeature(feature string) *FType {
	d := ch.Get(feature)
	if d == nil {
		return nil
	}
	return d.FT
}

func (ch *ChData) Name() string {
	return ch.name
}

func (ch *ChData) BatchSize() int {
	return ch.bs
}

func (ch *ChData) Describe(feature string, topK int) string {
	d := ch.Get(feature)
	if d == nil {
		return ""
	}

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

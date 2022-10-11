package seafan

import (
	"fmt"
	"github.com/invertedv/chutils"
	"io"
	"math/rand"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/stat"
)

// gdata.go implements structures and methods to produce gorgonia-ready data

type GDatum struct {
	FT      *FType  // FT stores the details of the field: it's role, # categories, mappings
	Summary Summary // Summary of the Data (e.g. distribution)
	Data    any     // Data. This will be either []float64 (FRCts, FROneHot, FREmbed) or []int32 (FRCat)
}

type GData struct {
	data          []*GDatum // data array
	dataRaw       []*Raw    // raw version
	rows          int       // # of observations in each GDatum
	sortField     string    // field data is sorted on (empty if not sorted)
	sortData      *GDatum   // *GDatum of sortField
	sortAscending bool      // sorts ascending, if true
	currRow       int       // current row for reader
}

// NewGData returns a new instance of GData
func NewGData() *GData {
	data := make([]*GDatum, 0)
	return &GData{data: data, rows: 0}
}

// Describe returns summary statistics. topK is # of values to return for discrete fields
func (g *GDatum) Describe(topK int) string {
	str := g.FT.String()

	switch g.FT.Role {
	case FRCts:
		str = fmt.Sprintf("%s%s", str, "\t"+strings.ReplaceAll(g.Summary.DistrC.String(), "\n", "\n\t"))
	case FRCat:
		str = fmt.Sprintf("%s\tTop 5 Values\n", str)
		str = fmt.Sprintf("%s%s", str, "\t"+strings.ReplaceAll(g.Summary.DistrD.TopK(topK, false, false), "\n", "\n\t"))
	}

	return str
}

func (g *GDatum) String() string {
	return g.Describe(0)
}

// check performs a sanity check on GData
func (gd *GData) check(name string) error {
	if name != "" {
		if gd.Get(name) != nil {
			return Wrapper(ErrGData, fmt.Sprintf("%s exists already", name))
		}
	}

	for _, d := range gd.data {
		if d.Summary.NRows != gd.rows {
			return Wrapper(ErrGData, "differing number of rows")
		}
	}

	return nil
}

// AppendC appends a continuous feature
func (gd *GData) AppendC(raw *Raw, name string, normalize bool, fp *FParam) error {
	if e := gd.check(name); e != nil {
		return e
	}

	x := make([]float64, len(raw.Data))

	for ind := 0; ind < len(x); ind++ {
		switch raw.Kind {
		case reflect.Float64:
			x[ind] = raw.Data[ind].(float64)
		case reflect.Float32:
			x[ind] = float64(raw.Data[ind].(float32))
		case reflect.Int:
			x[ind] = float64(raw.Data[ind].(int))
		case reflect.Int32:
			x[ind] = float64(raw.Data[ind].(int32))
		case reflect.Int64:
			x[ind] = float64(raw.Data[ind].(int64))
		case reflect.String:
			xx, e := strconv.ParseFloat(raw.Data[ind].(string), 64)
			if e != nil {
				return e
			}

			x[ind] = xx
		default:
			return Wrapper(ErrGData, fmt.Sprintf("AppendC: cannot convert this type %T", x[0]))
		}
	}

	ls := &FParam{}

	switch {
	case fp == nil:
		m, s := stat.MeanStdDev(x, nil)
		ls = &FParam{Location: m, Scale: s}
	case fp != nil:
		ls = fp
	}

	if normalize {
		if ls.Scale < 1e-8 {
			return Wrapper(ErrGData, fmt.Sprintf("AppendC: %s cannot be normalized--0 variance", name))
		}
		for ind := 0; ind < len(x); ind++ {
			x[ind] = (x[ind] - ls.Location) / ls.Scale
		}
	}

	distr, _ := NewDesc(nil, name)
	distr.Populate(x, true, nil)

	summ := Summary{
		NRows:  len(x),
		DistrC: distr,
		DistrD: nil,
	}
	ft := &FType{
		Name:       name,
		Role:       FRCts,
		Cats:       0,
		EmbCols:    0,
		Normalized: normalize,
		From:       "",
		FP:         ls,
	}
	c := &GDatum{
		Data:    x,
		FT:      ft,
		Summary: summ,
	}
	gd.data = append(gd.data, c)
	gd.rows = len(c.Data.([]float64))

	if e := gd.check(""); e != nil {
		return e
	}

	return nil
}

// AppendD appends a discrete feature
func (gd *GData) AppendD(raw *Raw, name string, fp *FParam) error {
	if e := gd.check(name); e != nil {
		return e
	}

	if fp == nil {
		lv := ByPtr(raw)
		fp = &FParam{Lvl: lv}
	}

	if raw.Kind == reflect.Float64 || raw.Kind == reflect.Float32 {
		return Wrapper(ErrGData, fmt.Sprintf("field %s cannot be FRCat (wrong type)", name))
	}

	ds := make([]int32, len(raw.Data))

	for ind := 0; ind < len(ds); ind++ {
		v := raw.Data[ind]
		val, ok := fp.Lvl[v]

		if !ok {
			val, ok = fp.Lvl[fp.Default]
			if !ok {
				return Wrapper(ErrGData, fmt.Sprintf("AppendD: default value %v not in dictionary", fp.Default))
			}
		}

		ds[ind] = val
	}

	distr := ByCounts(raw, nil)
	ft := &FType{
		Name:       name,
		Role:       FRCat,
		Cats:       len(fp.Lvl),
		EmbCols:    0,
		Normalized: false,
		From:       "",
		FP:         fp,
	}
	summ := Summary{
		NRows:  len(ds),
		DistrC: nil,
		DistrD: distr,
	}
	d := &GDatum{Data: ds, FT: ft, Summary: summ}
	gd.data = append(gd.data, d)
	gd.rows = len(d.Data.([]int32))

	if e := gd.check(""); e != nil {
		return e
	}

	return nil
}

// MakeOneHot creates & appends a one hot feature from a discrete feature
func (gd *GData) MakeOneHot(from, name string) error {
	if e := gd.check(name); e != nil {
		return e
	}

	d := gd.Get(from)

	if d == nil {
		return Wrapper(ErrGData, fmt.Sprintf("MakeOneHot: 'from' feature %s not found", from))
	}

	if d.FT.Role != FRCat {
		return Wrapper(ErrGData, fmt.Sprintf("MakeOneHot: input %s is not discrete", from))
	}

	nRow := d.Summary.NRows
	nCat := len(d.FT.FP.Lvl)
	oh := make([]float64, nRow*nCat)

	for row := 0; row < nRow; row++ {
		oh[int32(row*nCat)+d.Data.([]int32)[row]] = 1
	}

	summ := Summary{NRows: d.Summary.NRows}
	ft := &FType{
		Name:       name,
		Role:       FROneHot,
		Cats:       nCat,
		EmbCols:    0,
		Normalized: false,
		From:       from,
		FP:         nil,
	}
	oH := &GDatum{Data: oh, FT: ft, Summary: summ}
	gd.data = append(gd.data, oH)

	if e := gd.check(""); e != nil {
		return e
	}

	return nil
}

// Rows returns # of obserations in each element of GData
func (gd *GData) Rows() int {
	return gd.rows
}

// FieldCount returns the number of fields in GData
func (gd *GData) FieldCount() int {
	return len(gd.data)
}

// FieldList returns the names of the fields in GData
func (gd *GData) FieldList() []string {
	fl := make([]string, 0)
	for _, field := range gd.data {
		fl = append(fl, field.FT.Name)
	}

	return fl
}

// Get returns a single feature from GData
func (gd *GData) Get(name string) *GDatum {
	for _, g := range gd.data {
		if g.FT.Name == name {
			return g
		}
	}

	return nil
}

// Slice creates a new GData sliced according to sl
func (gd *GData) Slice(sl Slicer) (*GData, error) {

	if sl == nil {
		return gd, nil
	}

	gOut := NewGData()

	for _, g := range gd.data {
		ft := g.FT
		switch role := ft.Role; role {
		// These are all float64, but FROneHot and FREmbed are matrices
		case FRCts, FROneHot, FREmbed:
			cats := Max(1, ft.Cats)

			d := make([]float64, 0)
			n := 0
			for row := 0; row < g.Summary.NRows; row++ {
				if sl(row) {
					n++
					for r := 0; r < cats; r++ {
						d = append(d, g.Data.([]float64)[row*cats+r])
					}
				}
			}
			if len(d) == 0 {
				return nil, Wrapper(ErrGData, "slice result is empty")
			}
			var fp *FParam
			if ft.FP != nil {
				fp = &FParam{Location: ft.FP.Location, Scale: ft.FP.Scale}
			}
			ftNew := &FType{
				Name:       ft.Name,
				Role:       ft.Role,
				Cats:       ft.Cats,
				EmbCols:    ft.EmbCols,
				Normalized: ft.Normalized,
				From:       ft.From,
				FP:         fp,
			}
			desc, e := NewDesc(nil, ft.Name)
			if e != nil {
				return nil, Wrapper(e, fmt.Sprintf("Slice: error adding categorical field %s", ft.Name))
			}

			desc.Populate(d, true, nil)

			summ := Summary{
				NRows:  n,
				DistrC: desc,
				DistrD: nil,
			}
			datum := &GDatum{
				FT:      ftNew,
				Summary: summ,
				Data:    d,
			}
			gOut.data = append(gOut.data, datum)
			gOut.rows = n

		case FRCat:
			d := make([]int32, 0)
			for row := 0; row < g.Summary.NRows; row++ {
				if sl(row) {
					d = append(d, g.Data.([]int32)[row])
				}
			}

			if len(d) == 0 {
				return nil, Wrapper(ErrGData, "slice result is empty")
			}

			fp := &FParam{Lvl: ft.FP.Lvl, Default: ft.FP.Default}
			ftNew := &FType{
				Name:       ft.Name,
				Role:       ft.Role,
				Cats:       ft.Cats,
				EmbCols:    0,
				Normalized: false,
				From:       "",
				FP:         fp,
			}
			// This will be by the mapped value
			lvlsInt32 := ByCounts(NewRawCast(d, nil), nil)
			lvls := make(Levels)

			// keys array: inverse map of ft.FP.Lvl, so ith element is key that maps to i
			keys := make([]any, len(ft.FP.Lvl))
			for k, v := range ft.FP.Lvl {
				keys[v] = k
			}
			for k, v := range lvlsInt32 {
				lvls[keys[k.(int32)]] = v
			}

			summ := Summary{
				NRows:  len(d),
				DistrC: nil,
				DistrD: lvls,
			}
			datum := &GDatum{
				FT:      ftNew,
				Summary: summ,
				Data:    d,
			}
			gOut.rows = len(d)
			gOut.data = append(gOut.data, datum)
		}
	}
	if e := gOut.check(""); e != nil {
		return nil, Wrapper(e, "(*Gdata) Slice")
	}
	return gOut, nil
}

func (gd *GData) Swap(i, j int) {
	for ind := 0; ind < len(gd.data); ind++ {
		switch gd.data[ind].FT.Role {
		case FRCts:
			gd.data[ind].Data.([]float64)[i], gd.data[ind].Data.([]float64)[j] = gd.data[ind].Data.([]float64)[j], gd.data[ind].Data.([]float64)[i]
		case FRCat:
			gd.data[ind].Data.([]int32)[i], gd.data[ind].Data.([]int32)[j] = gd.data[ind].Data.([]int32)[j], gd.data[ind].Data.([]int32)[i]

		case FROneHot, FREmbed:
			cats := gd.data[ind].FT.Cats
			for c := 0; c < cats; c++ {
				gd.data[ind].Data.([]float64)[i*cats+c], gd.data[ind].Data.([]float64)[j*cats+c] =
					gd.data[ind].Data.([]float64)[j*cats+c], gd.data[ind].Data.([]float64)[i*cats+c]
			}
		}
	}
}

func (gd *GData) Len() int {
	return gd.rows
}

func (gd *GData) Less(i, j int) bool {
	switch gd.sortData.FT.Role {
	case FRCts:
		if gd.sortAscending {
			return gd.sortData.Data.([]float64)[i] < gd.sortData.Data.([]float64)[j]
		}
		return gd.sortData.Data.([]float64)[i] > gd.sortData.Data.([]float64)[j]
	case FRCat:
		if gd.sortAscending {
			return gd.sortData.Data.([]int32)[i] < gd.sortData.Data.([]int32)[j]
		}
		return gd.sortData.Data.([]int32)[i] > gd.sortData.Data.([]int32)[j]
	}

	return false
}

// Sort sorts the GData on field.  Calling Sort.Sort directly will cause a panic.
// Sorting a OneHot or Embedded field sorts on the underlying Categorical field
func (gd *GData) Sort(field string, ascending bool) error {
	defer func() { gd.sortData = nil }()

	gd.sortField = ""
	gd.sortAscending = ascending
	gDatum := gd.Get(field)
	if gDatum == nil {
		return Wrapper(ErrGData, fmt.Sprintf("(*GData) Sort: no such field %s", field))
	}

	// Sort on "From" field instead
	if gDatum.FT.Role == FROneHot || gDatum.FT.Role == FREmbed {
		if e := gd.Sort(gDatum.FT.From, ascending); e != nil {
			return e
		}
		gd.sortField = field
		return nil
	}

	gd.sortData = gDatum
	sort.Sort(gd)
	gd.sortField = field
	return nil
}

// IsSorted returns true if GData has been sorted by SortField
func (gd *GData) IsSorted() bool {
	return gd.sortField != ""
}

// SortField returns the field the GData is sorted on
func (gd *GData) SortField() string {
	return gd.sortField
}

// Shuffle shuffles the GData fields as a unit
func (gd *GData) Shuffle() {
	gd.sortField = ""

	rand.Seed(time.Now().UnixMicro())
	rand.Shuffle(gd.Len(), gd.Swap)
}

// GetRaw returns the raw data for the field.
func (gd *GData) GetRaw(field string) (*Raw, error) {
	fd := gd.Get(field)
	if fd == nil {
		return nil, Wrapper(ErrGData, fmt.Sprintf("(*GData) GetRaw: field %s not field", field))
	}
	switch fd.FT.Role {
	case FRCts:
		switch fd.FT.Normalized {
		case false:
			return NewRawCast(fd.Data.([]float64), nil), nil
		case true:
			x := make([]any, gd.rows)
			for ind := 0; ind < len(x); ind++ {
				x[ind] = fd.Data.([]float64)[ind]*fd.FT.FP.Scale + fd.FT.FP.Location
			}
			return NewRaw(x, nil), nil
		}
	case FRCat:
		key, _ := fd.FT.FP.Lvl.Sort(false, true)
		x := make([]any, gd.rows)
		for ind := 0; ind < len(x); ind++ {
			x[ind] = key[int(fd.Data.([]int32)[ind])]
		}
		return NewRaw(x, nil), nil
	case FROneHot, FREmbed:
		return gd.GetRaw(fd.FT.From)
	}
	return nil, Wrapper(ErrGData, "(*GData) GetRaw: unexpected error")
}

// UpdateFts produces a new *GData using the given FTypes.  The return only has those fields contained in newFts
func (gd *GData) UpdateFts(newFts FTypes) (*GData, error) {
	newGd := NewGData()
	newGd.rows = gd.rows

	for ind := 0; ind < len(gd.data); ind++ {
		oldFt := gd.data[ind].FT

		newFt := newFts.Get(oldFt.Name)
		// drop fields not in newFts
		if newFt == nil || newFt.Role == FROneHot || newFt.Role == FREmbed {
			continue
		}

		if newFt.Role != oldFt.Role {
			return nil, Wrapper(ErrGData, fmt.Sprintf("(*GData) UpdateFts: FRole differ for %s: %v (old) %v (new)",
				newFt.Name, oldFt.Role, newFt.Role))
		}

		// retrieve raw data to reprocess.
		raw, e := gd.GetRaw(oldFt.Name)
		if e != nil {
			return nil, e
		}

		switch newFt.Role {
		case FRCts:
			if e := newGd.AppendC(raw, newFt.Name, newFt.Normalized, newFt.FP); e != nil {
				return nil, e
			}
		case FRCat:
			if e := newGd.AppendD(raw, newFt.Name, newFt.FP); e != nil {
				return nil, e
			}
		}
	}

	for _, newFt := range newFts {
		if newFt.Role == FRCts || newFt.Role == FRCat {
			continue
		}

		if e := newGd.MakeOneHot(newFt.From, newFt.Name); e != nil {
			return nil, e
		}

		if newFt.Role == FREmbed {
			datum := newGd.Get(newFt.Name)
			datum.FT.Role = FREmbed
			datum.FT.EmbCols = newFt.EmbCols
		}
	}

	return newGd, nil
}

// Drop drops a field from *GData
func (gd *GData) Drop(field string) {
	newGd := make([]*GDatum, 0)
	for ind := 0; ind < len(gd.data); ind++ {
		if gd.data[ind].FT.Name != field {
			newGd = append(newGd, gd.data[ind])
		}
	}
	gd.data = newGd
}

// Read reads row(s) in the format of chutils.  Note: valids are all chutils.Valid.  Invoking Read for the first
// time causes it to recreate the raw data of existing fields -- so the memory requirement will go up.
func (gd *GData) Read(nTarget int, validate bool) (data []chutils.Row, valid []chutils.Valid, err error) {
	if nTarget <= 0 {
		return nil, nil, fmt.Errorf("(*GData) Read invalid nTarget")
	}

	// if this is the first read, then we need to populate the dataRaw array
	if gd.dataRaw == nil {
		gd.dataRaw = make([]*Raw, 0)
		for ind := 0; ind < len(gd.data); ind++ {
			datum := gd.data[ind]
			if datum.FT.Role == FREmbed || datum.FT.Role == FROneHot {
				continue
			}
			raw, e := gd.GetRaw(datum.FT.Name)
			if e != nil {
				return nil, nil, e
			}
			gd.dataRaw = append(gd.dataRaw, raw)
		}
	}

	data = make([]chutils.Row, 0)
	valid = make([]chutils.Valid, 0)

	for row := gd.currRow; row < gd.currRow+nTarget; row++ {
		gd.currRow = row
		rows := make(chutils.Row, 0)
		valids := make(chutils.Valid, 0)

		if row >= gd.rows {
			err = io.EOF
			gd.currRow = 0
			return
		}

		ind := 0
		for col := 0; col < len(gd.data); col++ {
			datum := gd.data[col]
			if datum.FT.Role == FREmbed || datum.FT.Role == FROneHot {
				continue
			}
			x := gd.dataRaw[ind].Data[row]
			rows = append(rows, x)
			ind++
		}

		data = append(data, rows)
		valid = append(valid, valids)
	}

	gd.currRow++ // set to next row
	return data, valid, err
}

func (gd *GData) CountLines() (numLines int, err error) {
	return gd.rows, nil
}

func (gd *GData) Reset() error {
	gd.currRow = 0
	return nil
}

func (gd *GData) Seek(lineNo int) error {
	if lineNo >= gd.rows {
		return chutils.Wrapper(chutils.ErrSeek, "seek past end of data")
	}

	if lineNo < 0 {
		return chutils.Wrapper(chutils.ErrSeek, "seek past beginning of data")
	}

	gd.currRow = lineNo

	return nil
}

func (gd *GData) Close() error {
	gd.currRow = 0
	return nil
}

func (gd *GData) TableSpec() *chutils.TableDef {
	fds := make(map[int]*chutils.FieldDef)

	ind := 0
	for col := 0; col < len(gd.data); col++ {
		datum := gd.data[col]
		fd := &chutils.FieldDef{
			Name:        datum.FT.Name,
			ChSpec:      chutils.ChField{},
			Description: "",
			Legal:       nil,
			Missing:     nil,
			Default:     nil,
			Width:       0,
			Drop:        false,
		}

		switch datum.FT.Role {
		case FREmbed, FROneHot:
			continue
		case FRCts:
			fd.ChSpec.Base, fd.ChSpec.Length = chutils.ChFloat, 64
		case FRCat:
			x := datum.FT.FP.Lvl.FindValue(0)
			switch x.(type) {
			case int32:
				fd.ChSpec.Base, fd.ChSpec.Length = chutils.ChInt, 32
			case int64:
				fd.ChSpec.Base, fd.ChSpec.Length = chutils.ChInt, 64
			case string:
				fd.ChSpec.Base = chutils.ChString
			case time.Time:
				fd.ChSpec.Base = chutils.ChDate
			default:
				return nil
			}
		}

		fds[ind] = fd
		ind++
	}

	key := fds[0].Name
	td := chutils.NewTableDef(key, chutils.MergeTree, fds)

	if e := td.Check(); e != nil {
		return nil
	}

	return td
}

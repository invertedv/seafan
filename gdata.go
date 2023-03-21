package seafan

import (
	"fmt"
	"io"
	"math/rand"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/invertedv/chutils"

	"gonum.org/v1/gonum/stat"
)

// gdata.go implements structures and methods to produce gorgonia-ready data

type GDatum struct {
	FT      *FType  // FT stores the details of the field: it's role, # categories, mappings
	Summary Summary // Summary of the Data (e.g. distribution)
	Data    any     // Data. This will be either []float64 (FRCts, FROneHot, FREmbed) or []int32 (FRCat)
	Raw     *Raw
}

type GData struct {
	data          []*GDatum // data array
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
func (gd *GData) AppendC(raw *Raw, name string, normalize bool, fp *FParam, keepRaw bool) error {
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

	if keepRaw {
		c.Raw = raw
	}

	gd.data = append(gd.data, c)
	gd.rows = len(c.Data.([]float64))

	if e := gd.check(""); e != nil {
		return e
	}

	return nil
}

// AppendD appends a discrete feature
func (gd *GData) AppendD(raw *Raw, name string, fp *FParam, keepRaw bool) error {
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
				return Wrapper(ErrGData, fmt.Sprintf("AppendD: default value %v not in dictionary, field %s", fp.Default, name))
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

	if keepRaw {
		d.Raw = raw
	}

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

			// if *Raw data isn't nil, must swap it, too
			if gd.data[ind].Raw != nil {
				gd.data[ind].Raw.Data[i], gd.data[ind].Raw.Data[j] = gd.data[ind].Raw.Data[j], gd.data[ind].Raw.Data[i]
			}
		case FRCat:
			gd.data[ind].Data.([]int32)[i], gd.data[ind].Data.([]int32)[j] = gd.data[ind].Data.([]int32)[j], gd.data[ind].Data.([]int32)[i]

			if gd.data[ind].Raw != nil {
				gd.data[ind].Raw.Data[i], gd.data[ind].Raw.Data[j] = gd.data[ind].Raw.Data[j], gd.data[ind].Raw.Data[i]
			}
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

// GetData returns the slice of *GDatums
func (gd *GData) GetData() []*GDatum {
	return gd.data
}

// GetRaw returns the raw data for the field.
func (gd *GData) GetRaw(field string) (*Raw, error) {
	fd := gd.Get(field)
	if fd == nil {
		return nil, fmt.Errorf("field %s not found", field)
	}

	if fd.Raw != nil {
		return fd.Raw, nil
	}

	if fd == nil {
		return nil, Wrapper(ErrGData, fmt.Sprintf("(*GData) GetRaw: field %s not field", field))
	}
	switch fd.FT.Role {
	case FRCts:
		switch fd.FT.Normalized {
		case false:
			fd.Raw = NewRawCast(fd.Data.([]float64), nil)
		case true:
			x := make([]any, gd.rows)
			for ind := 0; ind < len(x); ind++ {
				x[ind] = fd.Data.([]float64)[ind]*fd.FT.FP.Scale + fd.FT.FP.Location
			}
			fd.Raw = NewRaw(x, nil)
		}
	case FRCat:
		key, _ := fd.FT.FP.Lvl.Sort(false, true)
		x := make([]any, gd.rows)
		for ind := 0; ind < len(x); ind++ {
			x[ind] = key[int(fd.Data.([]int32)[ind])]
		}
		fd.Raw = NewRaw(x, nil)
	case FROneHot, FREmbed:
		return gd.GetRaw(fd.FT.From)
	}

	return fd.Raw, nil
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
			if e := newGd.AppendC(raw, newFt.Name, newFt.Normalized, newFt.FP, false); e != nil {
				return nil, e
			}
		case FRCat:
			if e := newGd.AppendD(raw, newFt.Name, newFt.FP, false); e != nil {
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
func (gd *GData) Drop(field string) error {
	newGd := make([]*GDatum, 0)
	ok := false
	for ind := 0; ind < len(gd.data); ind++ {
		if gd.data[ind].FT.Name == field {
			ok = true
		}

		if gd.data[ind].FT.Name != field {
			newGd = append(newGd, gd.data[ind])
		}
	}
	if !ok {
		return fmt.Errorf("field %s not found", field)
	}

	gd.data = newGd

	return nil
}

// Keep drops all fields not in "fields"
func (gd *GData) Keep(fields []string) error {
	newGd := make([]*GDatum, 0)

	for ind := 0; ind < len(fields); ind++ {
		gdatum := gd.Get(fields[ind])

		if gdatum == nil {
			return fmt.Errorf("field not found: %s, (*GData) Keep", fields[ind])
		}

		newGd = append(newGd, gdatum)
	}

	gd.data = newGd

	return nil
}

// Read reads row(s) in the format of chutils.  Note: valids are all chutils.Valid.  Invoking Read for the first
// time causes it to recreate the raw data of existing fields -- so the memory requirement will go up.
func (gd *GData) Read(nTarget int, validate bool) (data []chutils.Row, valid []chutils.Valid, err error) {
	if nTarget <= 0 {
		return nil, nil, fmt.Errorf("(*GData) Read invalid nTarget")
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
			var e error
			datum := gd.data[col]
			if datum.Raw == nil {
				datum.Raw, e = gd.GetRaw(datum.FT.Name)
				if e != nil {
					return nil, nil, e
				}
			}
			if datum.FT.Role == FREmbed || datum.FT.Role == FROneHot {
				continue
			}
			x := gd.data[ind].Raw.Data[row]
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

// AppendField adds a field to gd
func (gd *GData) AppendField(newData *Raw, name string, fRole FRole, keepRaw bool) error {
	// drop field if it's already there
	_ = gd.Drop(name)

	switch fRole {
	case FRCts:
		if e := gd.AppendC(newData, name, false, nil, keepRaw); e != nil {
			return e
		}
	case FRCat, FROneHot, FREmbed:
		if e := gd.AppendD(newData, name, nil, keepRaw); e != nil {
			return e
		}
	}

	if fRole == FROneHot || fRole == FREmbed {
		if e := gd.MakeOneHot(name, name+"Oh"); e != nil {
			return e
		}
	}

	return nil
}

// Back2Raw converts the entire GData back to its raw state
func (gd *GData) Back2Raw() (rawData []*Raw, nCol int, fields []string, err error) {
	fields = gd.FieldList()
	nCol = len(fields)
	rawData = make([]*Raw, nCol)

	for cols := 0; cols < nCol; cols++ {
		var e error
		rawData[cols], e = gd.GetRaw(fields[cols])
		if e != nil {
			return nil, 0, nil, e
		}
	}

	return rawData, nCol, fields, nil
}

func (gd *GData) Row(take int) (gdNew *GData, err error) {
	if take < 0 || take >= gd.Rows() {
		return nil, fmt.Errorf("row out of range (*GData)Row: %d", take)
	}

	gdNew = NewGData()

	for ind, fld := range gd.FieldList() {
		var rawBig *Raw
		if rawBig, err = gd.GetRaw(fld); err != nil {
			return nil, err
		}

		raw := NewRaw([]any{rawBig.Data[take]}, nil)
		datum := gd.data[ind]

		switch datum.FT.Role {
		case FRCat:
			err = gdNew.AppendD(raw, datum.FT.Name, datum.FT.FP, datum.Raw != nil)
		case FRCts, FREither:
			err = gdNew.AppendC(raw, datum.FT.Name, datum.FT.Normalized, datum.FT.FP, datum.Raw != nil)
		case FROneHot, FREmbed:
			err = gdNew.MakeOneHot(datum.FT.From, datum.FT.Name)
		}

		if err != nil {
			return nil, err
		}
	}

	return gdNew, nil
}

// Subset subsets the pipeline to the rows in keepRows
func (gd *GData) Subset(keepRows []int) (gdOut *GData, err error) {
	flds := gd.FieldList()
	gdOut = NewGData()

	for ind := 0; ind < len(flds); ind++ {
		var (
			raw  *Raw
			e    error
			data []any
		)

		raw, e = gd.GetRaw(flds[ind])
		if e != nil {
			return nil, e
		}
		datum := gd.data[ind]

		for indx := 0; indx < len(keepRows); indx++ {
			indKeep := keepRows[indx]
			if indKeep >= len(raw.Data) || indKeep < 0 {
				return nil, fmt.Errorf("index out of range: %d to array of length %d", indKeep, len(raw.Data))
			}

			data = append(data, raw.Data[indKeep])
		}

		rawNew := NewRaw(data, nil)

		switch datum.FT.Role {
		case FRCat:
			e = gdOut.AppendD(rawNew, datum.FT.Name, datum.FT.FP, datum.Raw != nil)
		case FRCts, FREither:
			e = gdOut.AppendC(rawNew, datum.FT.Name, datum.FT.Normalized, datum.FT.FP, datum.Raw != nil)
		case FROneHot, FREmbed:
			e = gdOut.MakeOneHot(datum.FT.From, datum.FT.Name)
		}

		if e != nil {
			return nil, e
		}
	}

	return gdOut, nil
}

func (gd *GData) Where(field string, equalTo []any) (gdOut *GData, err error) {
	var raw *Raw

	if raw, err = gd.GetRaw(field); err != nil {
		return nil, err
	}

	var rows []int

	for ind := 0; ind < raw.Len(); ind++ {
		match := false
		switch x := raw.Data[ind].(type) {
		case float32:
			for _, eq := range equalTo {
				if y, ok := eq.(float32); ok {
					if match = x == y; match {
						break
					}
				}
			}
		case float64:
			for _, eq := range equalTo {
				if y, ok := eq.(float64); ok {
					if match = x == y; match {
						break
					}
				}
			}
		case int32:
			for _, eq := range equalTo {
				if y, ok := eq.(int32); ok {
					if match = x == y; match {
						break
					}
				}
			}
		case int64:
			for _, eq := range equalTo {
				if y, ok := eq.(int64); ok {
					if match = x == y; match {
						break
					}
				}
			}
		case string:
			for _, eq := range equalTo {
				if y, ok := eq.(string); ok {
					if match = x == y; match {
						break
					}
				}
			}
		case time.Time:
			for _, eq := range equalTo {
				if y, ok := eq.(time.Time); ok {
					if match = x.Sub(y) == 0; match {
						break
					}
				}
			}
		}

		if match {
			rows = append(rows, ind)
		}
	}

	if rows == nil {
		return nil, fmt.Errorf("no matches in Where")
	}

	return gd.Subset(rows)
}

// AppendRowsRaw simply appends rows, in place, to the existing GData.  Only the *Raw data is updated.
// The .Data field is set to nil.
func (gd *GData) AppendRowsRaw(gdApp *GData) error {
	for ind, fld := range gd.FieldList() {
		rawApp, e := gdApp.GetRaw(fld)
		if e != nil {
			return e
		}

		if gd.data[ind].Raw == nil {
			raw, e := gd.GetRaw(fld)
			if e != nil {
				return e
			}
			gd.data[ind].Raw = raw
		}
		gd.data[ind].Raw.Data = append(gd.data[ind].Raw.Data, rawApp.Data...)
		gd.data[ind].Data = nil
	}

	gd.rows += gdApp.rows

	return nil
}

// AppendRows appends rows to the existing GData and then re-initializes each GDatum, using the fTypes, if provided.
func (gd *GData) AppendRows(gdApp *GData, fTypes FTypes) (gdOut *GData, err error) {
	gdOut = NewGData()
	for ind, fld := range gd.FieldList() {
		rawApp, e := gdApp.GetRaw(fld)
		if e != nil {
			return nil, e
		}

		hasRaw := true
		if gd.data[ind].Raw == nil {
			hasRaw = false
			raw, ex := gd.GetRaw(fld)
			if ex != nil {
				return nil, ex
			}
			gd.data[ind].Raw = raw
		}
		dataOut := make([]any, gd.data[ind].Raw.Len())

		ft := gd.data[ind].FT
		// if fTypes is nil, set fp to nil so FP will be recalculated
		var fp *FParam = nil
		if fTypes != nil {
			if ftApp := fTypes.Get(fld); ftApp != nil {
				ft = ftApp
				fp = ftApp.FP
			}
		}

		copy(dataOut, gd.data[ind].Raw.Data)
		dataOut = append(dataOut, rawApp.Data...)
		rawNew := NewRaw(dataOut, nil)

		switch ft.Role {
		case FRCat:
			e = gdOut.AppendD(rawNew, ft.Name, fp, hasRaw)
		case FRCts, FREither:
			e = gdOut.AppendC(rawNew, ft.Name, ft.Normalized, fp, hasRaw)
		case FROneHot, FREmbed:
			e = gdOut.MakeOneHot(ft.From, ft.Name)
		}

		if e != nil {
			return nil, e
		}
	}

	return gdOut, nil
}

// ReInit re-initializes the Data field from Raw for each GDatum. If ftypes is not nil, these values
// are used, otherwise the FParam values are re-derived from the data.
func (gd *GData) ReInit(fTypes *FTypes) (gdOut *GData, err error) {
	gdOut = NewGData()

	for ind, fld := range gd.FieldList() {
		var (
			ft *FType
			fp *FParam
		)

		if fTypes != nil {
			ft = fTypes.Get(fld)
			fp = ft.FP
		}

		if ft == nil {
			ft = gd.data[ind].FT
		}

		var rawData *Raw
		rawData, err = gd.GetRaw(fld)

		switch ft.Role {
		case FRCat:
			err = gdOut.AppendD(rawData, ft.Name, fp, true)
		case FRCts, FREither:
			err = gdOut.AppendC(rawData, ft.Name, ft.Normalized, fp, true)
		case FROneHot, FREmbed:
			err = gdOut.MakeOneHot(ft.From, ft.Name)
		}

		if err != nil {
			return nil, err
		}
	}

	return gdOut, nil
}

// GetFTypes returns a slice of *FType corresponding to GData.data
func (gd *GData) GetFTypes() FTypes {
	fts := make(FTypes, 0)
	for _, d := range gd.data {
		fts = append(fts, d.FT)
	}

	return fts
}

// Get FType returns the *FType of field.  Returns
func (gd *GData) GetFType(field string) *FType {
	for _, d := range gd.data {
		if d.FT.Name == field {
			return d.FT
		}
	}

	return nil
}

// JoinType is the method to use in joining two GData structs
//
//go:generate stringer -type=JoinType
type JoinType int

const (
	Inner JoinType = 0 + iota
	Left
	Right
	Outer
)

func joinCheck(left, right *Raw, onField string) error {
	if left.Kind == reflect.Float32 || left.Kind == reflect.Float64 {
		return fmt.Errorf("cannot join on Float")
	}

	if right.Kind == reflect.Float32 || right.Kind == reflect.Float64 {
		return fmt.Errorf("cannot join on Float")
	}

	if left.Kind != right.Kind {
		return fmt.Errorf("join types not the same")
	}

	return nil
}

// Join joins two *GData on onField.
// Both *GData are sorted by onField, though the result may not be in sort order for Outer and Right joins.
// If a field value is missing, the FType.FParam.Default value is filled in. If that value is nil, the following
// are used:
//   - int,float : 0
//   - string ""
//   - time.Time: 1/1/1970
func (gd *GData) Join(right *GData, onField string, joinType JoinType) (result *GData, err error) {
	var (
		ulRaw, urRaw, lRaw, rRaw             []*Raw
		lJoin, rJoin                         *Raw
		lFts, rFts                           FTypes
		ulFields, urFields, lFields, rFields []string
	)

	if right == nil {
		return nil, fmt.Errorf("right *GDatais nil")
	}

	if lJoin, err = gd.GetRaw(onField); err != nil {
		return nil, err
	}

	if rJoin, err = right.GetRaw(onField); err != nil {
		return nil, err
	}

	if gd.sortField != onField || !gd.sortAscending {
		if e := gd.Sort(onField, true); e != nil {
			return nil, e
		}
	}

	if right.sortField != onField || !right.sortAscending {
		if e := right.Sort(onField, true); e != nil {
			return nil, e
		}
	}

	if e := joinCheck(lJoin, rJoin, onField); e != nil {
		return nil, e
	}

	if ulRaw, _, ulFields, err = gd.Back2Raw(); err != nil {
		return nil, err
	}
	if urRaw, _, urFields, err = right.Back2Raw(); err != nil {
		return nil, err
	}

	ulFts, urFts := gd.GetFTypes(), right.GetFTypes()

	// subset to fields to keep
	lFields, lRaw, lFts = subsetFields(ulFields, ulRaw, ulFts, []string{onField})

	omit := []string{onField}

	for _, fld := range urFields {
		if searchSlice(fld, lFields) >= 0 {
			omit = append(omit, fld)
		}
	}

	rFields, rRaw, rFts = subsetFields(urFields, urRaw, urFts, omit)

	lInd, rInd := 0, 0
	lResult, rResult := make([][]any, len(lFields)), make([][]any, len(rFields))
	joinResult := make([]any, 0)

	for {
		// list of all indices that equal jl.Data[lInd]
		lEqual, rEqual, e := collectEqual(lJoin, rJoin, lInd, rInd)
		if e != nil {
			return nil, e
		}

		if rEqual != nil {
			// exact matches always join
			lResult, rResult, joinResult = collectResults(lRaw, rRaw, rFts, lEqual, rEqual, lResult, rResult, lJoin, joinResult)

			// Did we skip rows on right?
			if (joinType == Right || joinType == Outer) && rEqual[0] > rInd {
				// Add all missing rows
				var rTake []int
				for ind := rInd; ind < rEqual[0]; ind++ {
					rTake = append(rTake, ind)
				}

				rResult, lResult, joinResult = collectResults(rRaw, lRaw, lFts, rTake, nil, rResult, lResult, rJoin, joinResult)
			}

			rInd = rEqual[len(rEqual)-1] + 1
		}

		// if left or outer join, add unmatched left values
		if (joinType == Left || joinType == Outer) && rEqual == nil {
			lResult, rResult, joinResult = collectResults(lRaw, rRaw, rFts, lEqual, nil, lResult, rResult, lJoin, joinResult)
		}

		if lEqual[len(lEqual)-1] == gd.Rows()-1 {
			break
		}

		lInd = lEqual[len(lEqual)-1] + 1
	}

	// Did we leave some right rows behind?
	if (joinType == Right || joinType == Outer) && rInd < right.Len() {
		// Add all missing rows
		var rTake []int
		for ind := rInd; ind < right.Len(); ind++ {
			rTake = append(rTake, ind)
		}

		rResult, lResult, joinResult = collectResults(rRaw, lRaw, lFts, rTake, nil, rResult, lResult, rJoin, joinResult)
	}

	result = NewGData()
	if e := result.AddRaw(lResult, lFields, lFts, true); e != nil {
		return nil, e
	}

	if e := result.AddRaw(rResult, rFields, rFts, true); e != nil {
		return nil, e
	}

	if e := result.AppendD(NewRaw(joinResult, nil), onField, nil, true); e != nil {
		return nil, e
	}

	return result, nil
}

// Adds a number of fields to a *GData. The fts are only used to determine the Role
func (gd *GData) AddRaw(data [][]any, fields []string, fts FTypes, keepRaw bool) error {
	for ind := 0; ind < len(fields); ind++ {
		raw := NewRaw(data[ind], nil)
		switch fts[ind].Role {
		case FRCat:
			if e := gd.AppendD(raw, fields[ind], nil, keepRaw); e != nil {
				return e
			}
		case FRCts, FREither:
			if e := gd.AppendC(raw, fields[ind], false, nil, keepRaw); e != nil {
				return e
			}
		case FROneHot, FREmbed:
			if e := gd.MakeOneHot(fts[ind].From, fts[ind].Name); e != nil {
				return e
			}

		}
	}

	return nil
}

// getMiss gets/sets the default value of a field
func getMiss(ft *FType, kind reflect.Kind) any {
	if ft.FP.Default != nil {
		return ft.FP.Default
	}

	switch kind {
	case reflect.Float64:
		ft.FP.Default = float64(0)
	case reflect.Float32:
		ft.FP.Default = float32(0)
	case reflect.Int32:
		ft.FP.Default = int32(0)
	case reflect.Int64:
		ft.FP.Default = int64(0)
	case reflect.String:
		ft.FP.Default = ""
	case reflect.Struct:
		ft.FP.Default = time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC)
	}

	return ft.FP.Default
}

// subsets a *Raw slice to drop the omit fields
func subsetFields(uFields []string, uRaw []*Raw, uFts FTypes, omit []string) (fields []string, raw []*Raw, fts FTypes) {
	for ind := 0; ind < len(uFields); ind++ {
		fld := uFields[ind]
		if searchSlice(fld, omit) >= 0 {
			continue
		}

		fields = append(fields, fld)
		raw = append(raw, uRaw[ind])
		fts = append(fts, uFts[ind])
	}

	return fields, raw, fts
}

// equalInd returns a slice of indices where data.Data have the same value as data.Data[startInd]
func equalInd(data *Raw, startInd int) (inds []int, err error) {
	var comp bool
	for ind := startInd; ind < data.Len(); ind++ {
		if comp, err = Comparer(data.Data[startInd], data.Data[ind], "=="); err != nil {
			return nil, err
		}

		if !comp {
			break
		}

		inds = append(inds, ind)
	}

	return inds, nil
}

// collectEqual returns a list of indices in left & right that have the same value as left.Data[lInd].
//   - left, right are join *Raw data values
//   - lInd is the key that determines the value we are looking for
//   - rInd is the starting point in right for the search
func collectEqual(left, right *Raw, lInd, rInd int) (lEqual, rEqual []int, err error) {
	var comp bool
	// out of right rows? if so, return remainder of left
	if rInd == right.Len() {
		for ind := lInd; ind < left.Len(); ind++ {
			lEqual = append(lEqual, ind)
		}

		return lEqual, nil, nil
	}

	// find all on left side that equal our first element
	if lEqual, err = equalInd(left, lInd); err != nil {
		return nil, nil, err
	}

	if comp, err = Comparer(left.Data[lInd], right.Data[rInd], "<"); err != nil {
		return nil, nil, err
	}

	// left less than next right?
	if comp {
		return lEqual, nil, nil
	}

	// find first equal
	rEq := -1
	for ind := rInd; ind < right.Len(); ind++ {
		if comp, err = Comparer(left.Data[lInd], right.Data[ind], "=="); err != nil {
			return nil, nil, err
		}
		if comp {
			rEq = ind
			break
		}
	}

	// none equal
	if rEq == -1 {
		return lEqual, nil, nil
	}
	if rEqual, err = equalInd(right, rEq); err != nil {
		return nil, nil, err
	}

	return lEqual, rEqual, nil
}

// collectResults adds rows to lResult, rResult and joinResult returning them as lUp, rUp, joinUp.
//   - left and right are th
func collectResults(left, right []*Raw, rFts FTypes, lEqual, rEqual []int, lResult, rResult [][]any,
	joinRaw *Raw, joinResult []any) (lUp, rUp [][]any, joinUp []any) {
	//

	joinUp = joinResult
	lUp = lResult
	rUp = rResult

	for _, indL := range lEqual {
		if rEqual == nil {
			joinUp = append(joinUp, joinRaw.Data[indL])

			for col := 0; col < len(left); col++ {
				lUp[col] = append(lUp[col], left[col].Data[indL])
			}

			for col := 0; col < len(right); col++ {
				miss := getMiss(rFts[col], right[col].Kind)
				rUp[col] = append(rUp[col], miss)
			}

			continue
		}

		for _, indR := range rEqual {
			joinUp = append(joinUp, joinRaw.Data[indL])

			for col := 0; col < len(left); col++ {
				lUp[col] = append(lUp[col], left[col].Data[indL])
			}

			for col := 0; col < len(right); col++ {
				rUp[col] = append(rUp[col], right[col].Data[indR])
			}
		}
	}

	return lUp, rUp, joinUp
}

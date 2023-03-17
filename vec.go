package seafan

import (
	"fmt"
	"reflect"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type VecData struct {
	bs         int    // batch size
	cbRow      int    // current batch starting row
	nRow       int    // # rows in dataset
	data       *GData // processed data
	epochCount int    // current epoch
	ftypes     FTypes // user input selections
	callback   Opts   // user callbacks executed at the start of Init()
	keepRaw    bool   // if true, *Raw data is retained
	name       string // pipeline name
}

func NewVecData(name string, data *GData, opts ...Opts) *VecData {
	vec := &VecData{bs: 1, data: data, name: name}
	vec.nRow = vec.data.data[0].Summary.NRows

	for _, gd := range vec.data.data {
		vec.ftypes = append(vec.ftypes, gd.FT)
	}
	for _, o := range opts {
		o(vec)
	}

	return vec
}

// VecFromAny builds a pipeline for a slice of vectors ([]any).  The first dimension is the field.
func VecFromAny(data [][]any, fields []string, ftypes FTypes) (pipe Pipeline, err error) {
	gd := NewGData()
	for ind, field := range fields {
		raw := NewRaw(data[ind], nil)

		role := FRCts
		if raw.Kind == reflect.String || raw.Kind == reflect.Struct {
			role = FRCat
		}

		if ft := ftypes.Get(field); ft != nil {
			if role != FRCat && role != FRCts {
				return nil, fmt.Errorf("must be FRCat or FRCts, field %s is not VecFromAny", field)
			}
			role = ft.Role
		}

		if role == FRCat {
			if e := gd.AppendD(raw, field, nil, true); e != nil {
				return nil, e
			}
			continue
		}

		if e := gd.AppendC(raw, field, false, nil, true); e != nil {
			return nil, e
		}
	}

	pipe = NewVecData("values", gd)

	return pipe, nil
}

func (vec *VecData) GetKeepRaw() bool {
	return vec.keepRaw
}

func (vec *VecData) Slice(sl Slicer) (Pipeline, error) {
	gOut, e := vec.data.Slice(sl)

	if e != nil {
		return nil, Wrapper(e, "*VecData.Slice")
	}
	name := fmt.Sprintf("sliced from %s", vec.name)
	return NewVecData(name, gOut), nil
}

func (vec *VecData) Init() error {
	vec.cbRow = 0
	if vec.bs == 0 {
		vec.bs = vec.Rows()
	}

	return nil
}

func (vec *VecData) Batch(inputs G.Nodes) bool {
	// out of data?  if NRows % bsize !=0, rows after the last full batvec are unused.
	if vec.cbRow+vec.bs > vec.nRow {
		vec.cbRow = 0
		// user callbacks
		if vec.callback != nil {
			vec.callback(vec)
		}
		return false
	}

	startRow := vec.cbRow
	endRow := startRow + vec.bs

	for _, nd := range inputs {
		var t tensor.Tensor

		d := vec.data.Get(nd.Name())

		if d == nil {
			panic(Wrapper(ErrVecData, fmt.Sprintf("feature %s not in dataset", nd.Name())))
		}

		switch d.FT.Role {
		case FRCts:
			t = tensor.New(tensor.WithBacking(d.Data.([]float64)[startRow:endRow]), tensor.WithShape(vec.bs, 1))
		case FRCat:
			t = tensor.New(tensor.WithBacking(d.Data.([]int32)[startRow:endRow]), tensor.WithShape(vec.bs, 1))
		case FROneHot, FREmbed:
			sr := startRow * d.FT.Cats
			er := endRow * d.FT.Cats
			t = tensor.New(tensor.WithBacking(d.Data.([]float64)[sr:er]), tensor.WithShape(vec.bs, d.FT.Cats))
		}

		if e := G.Let(nd, t); e != nil {
			panic(e)
		}
	}

	vec.cbRow = endRow

	return true
}

// Rows is # of rows of data in the Pipeline
func (vec *VecData) Rows() int {
	return vec.nRow
}

// GetFTypes returns FTypes for vec Pipeline.
func (vec *VecData) GetFTypes() FTypes {
	fts := make(FTypes, 0)
	for _, d := range vec.data.data {
		fts = append(fts, d.FT)
	}

	return fts
}

// SaveFTypes saves the FTypes for the Pipeline.
func (vec *VecData) SaveFTypes(fileName string) error {
	return Wrapper(vec.GetFTypes().Save(fileName), "(*VecData).SaveFTypes")
}

// IsNormalized returns true if the field is normalized.
func (vec *VecData) IsNormalized(field string) bool {
	if ft := vec.Get(field); ft != nil {
		return ft.FT.Normalized
	}

	return false
}

// IsCts returns true if the field has role FRCts.
func (vec *VecData) IsCts(field string) bool {
	if ft := vec.Get(field); ft != nil {
		return ft.FT.Role == FRCat
	}

	return false
}

// IsCat returns true if field has role FRCat.
func (vec *VecData) IsCat(field string) bool {
	if ft := vec.Get(field); ft != nil {
		return ft.FT.Role == FRCat
	}

	return false
}

// GData returns the Pipelines' GData
func (vec *VecData) GData() *GData {
	d := vec.data

	return d
}

// Get returns a fields's GDatum
func (vec *VecData) Get(field string) *GDatum {
	return vec.data.Get(field)
}

// Cols returns the # of columns in the field
func (vec *VecData) Cols(field string) int {
	d := vec.Get(field)

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

// Epoch sets the epoch to setTo if setTo >=0 and returns epoch #.
func (vec *VecData) Epoch(setTo int) int {
	if setTo >= 0 {
		vec.epochCount = setTo
	}

	return vec.epochCount
}

// FieldList returns a slice of field names in the Pipeline
func (vec *VecData) FieldList() []string {
	fl := make([]string, 0)
	for _, ft := range vec.data.data {
		fl = append(fl, ft.FT.Name)
	}

	return fl
}

// GetFType returns the fields FType
func (vec *VecData) GetFType(field string) *FType {
	d := vec.Get(field)
	if d == nil {
		return nil
	}

	return d.FT
}

// Name returns Pipeline name
func (vec *VecData) Name() string {
	return vec.name
}

// BatchSize returns Pipeline batch size
func (vec *VecData) BatchSize() int {
	return vec.bs
}

// Describe describes a field.  If the field has role FRCat, the top k values (by frequency) are returned.
func (vec *VecData) Describe(field string, topK int) string {
	d := vec.Get(field)
	if d == nil {
		return ""
	}

	return d.Describe(topK)
}

func (vec *VecData) String() string {
	const numCats = 5
	str := fmt.Sprintf("Summary for pipeline %s\n", vec.Name())
	fl := vec.FieldList()
	str = fmt.Sprintf("%s%d fields\n", str, len(fl))

	for _, f := range fl {
		ff := vec.Describe(f, numCats)
		str += ff
	}

	return str
}

// Shuffle shuffles the data.
func (vec *VecData) Shuffle() {
	vec.data.Shuffle()
}

// Sort sorts the data on "field".
func (vec *VecData) Sort(field string, ascending bool) error {
	e := vec.data.Sort(field, ascending)
	if e != nil {
		return Wrapper(e, "(*ChData) Sort")
	}
	return nil
}

// IsSorted returns true if the data has been sorted.
func (vec *VecData) IsSorted() bool {
	return vec.data.IsSorted()
}

// SortField returns the name of the sort field.
func (vec *VecData) SortField() string {
	return vec.data.SortField()
}

func (vec *VecData) Row(take int) (newPipe Pipeline, err error) {
	var gdNew *GData

	if gdNew, err = vec.GData().Row(take); err != nil {
		return nil, err
	}

	return NewVecData("new pipe", gdNew), nil
}

func (vec *VecData) FieldCount() int {
	return vec.data.FieldCount()
}

func (vec *VecData) Subset(rows []int) (newPipe Pipeline, err error) {
	var gdNew *GData

	if gdNew, err = vec.GData().Subset(rows); err != nil {
		return nil, err
	}

	return NewVecData("new pipe", gdNew), nil
}

func (vec *VecData) Where(field string, equalTo []any) (newPipe Pipeline, err error) {
	var gdNew *GData

	if gdNew, err = vec.GData().Where(field, equalTo); err != nil {
		return nil, err
	}

	return NewVecData("new pipe", gdNew), nil
}

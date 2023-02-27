package seafan

// pipeline.go has the interface and "With" funcs for Pipelines.
import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/invertedv/chutils"
	cf "github.com/invertedv/chutils/file"
	s "github.com/invertedv/chutils/sql"
	G "gorgonia.org/gorgonia"
)

// The Pipeline interface specifies the methods required to be a data Pipeline. The Pipeline is the middleware between
// the data and the fitting routines.
type Pipeline interface {
	Init() error                            // initialize the pipeline
	Rows() int                              // # of observations in the pipeline (size of the epoch)
	Batch(inputs G.Nodes) bool              // puts the next batch in the input nodes
	Epoch(setTo int) int                    // manage epoch count
	IsNormalized(field string) bool         // true if feature is normalized
	IsCat(field string) bool                // true if feature is one-hot encoded
	Cols(field string) int                  // # of columns in the feature
	IsCts(field string) bool                // true if the feature is continuous
	GetFType(field string) *FType           // Get FType for the feature
	GetFTypes() FTypes                      // Get Ftypes for pipeline
	BatchSize() int                         // batch size
	FieldList() []string                    // fields available
	GData() *GData                          // return underlying GData
	Get(field string) *GDatum               // return data for field
	Slice(sl Slicer) (Pipeline, error)      // slice the pipeline
	Shuffle()                               // shuffle data
	Describe(field string, topK int) string // describes a field
}

// Opts function sets an option to a Pipeline
type Opts func(c Pipeline)

// WithBatchSize sets the batch size for the pipeline
func WithBatchSize(bsize int) Opts {
	f := func(c Pipeline) {
		if bsize == 0 {
			bsize = c.Rows()
		}
		switch d := c.(type) {
		case *ChData:
			d.bs = bsize
		case *VecData:
			d.bs = bsize
		}
	}

	return f
}

// WithCycle sets the cycle bool.  If false, the intent is for the Pipeline to generate a new
// data set is generated for each epoch.
func WithCycle(cycle bool) Opts {
	f := func(c Pipeline) {
		switch d := c.(type) {
		case *ChData:
			d.cycle = cycle
		}
	}

	return f
}

// WithCats specifies a list of categorical features.
func WithCats(names ...string) Opts {
	f := func(c Pipeline) {
		switch d := c.(type) {
		case *ChData:
			for _, nm := range names {
				ft := d.ftypes.Get(nm)
				if ft != nil {
					ft.Role = FRCat

					continue
				}

				ft = &FType{
					Name: nm,
					Role: FRCat,
				}
				d.ftypes = append(d.ftypes, ft)
			}
		case *VecData:
			for _, nm := range names {
				ft := d.ftypes.Get(nm)
				if ft != nil {
					ft.Role = FRCat

					continue
				}

				ft = &FType{
					Name: nm,
					Role: FRCat,
				}
				d.ftypes = append(d.ftypes, ft)
			}
		}
	}

	return f
}

// WithOneHot adds a one-hot field "name" based of field "from"
func WithOneHot(name, from string) Opts {
	f := func(c Pipeline) {
		switch d := c.(type) {
		case *ChData:
			ft := d.ftypes.Get(name)
			if ft != nil {
				ft.From = from
				ft.Role = FROneHot

				return
			}

			ft = &FType{
				Name: name,
				Role: FROneHot,
				From: from,
			}
			d.ftypes = append(d.ftypes, ft)
		case *VecData:
			ft := d.ftypes.Get(name)
			if ft != nil {
				ft.From = from
				ft.Role = FROneHot

				return
			}

			ft = &FType{
				Name: name,
				Role: FROneHot,
				From: from,
			}
			d.ftypes = append(d.ftypes, ft)
		}
	}

	return f
}

// WithNormalized sets the features to be normalized.
func WithNormalized(names ...string) Opts {
	f := func(c Pipeline) {
		switch d := c.(type) {
		case *ChData:
			for _, nm := range names {
				ft := d.ftypes.Get(nm)
				if ft != nil {
					ft.Role = FRCts
					ft.Normalized = true

					continue
				}

				ft = &FType{
					Name:       nm,
					Role:       FRCts,
					Normalized: true,
				}
				d.ftypes = append(d.ftypes, ft)
			}
		case *VecData:
			for _, nm := range names {
				ft := d.ftypes.Get(nm)
				if ft != nil {
					ft.Role = FRCts
					ft.Normalized = true

					continue
				}

				ft = &FType{
					Name:       nm,
					Role:       FRCts,
					Normalized: true,
				}
				d.ftypes = append(d.ftypes, ft)
			}
		}
	}

	return f
}

// WithFtypes sets the FTypes of the Pipeline. The feature is used to override the default levels.
func WithFtypes(fts FTypes) Opts {
	f := func(c Pipeline) {
		switch d := c.(type) {
		case *ChData:
			for _, ft := range fts {
				ok := false
				// see if it's already there
				for ind := 0; ind < len(d.ftypes); ind++ {
					if d.ftypes[ind].Name == ft.Name {
						d.ftypes[ind] = ft
						ok = true
						break
					}
				}

				if !ok {
					d.ftypes = append(d.ftypes, ft)
				}
			}

			d.ftypes = fts
		case *VecData:
			for _, ft := range fts {
				ok := false
				for ind := 0; ind < len(d.ftypes); ind++ {
					if d.ftypes[ind].Name == ft.Name {
						d.ftypes[ind] = ft
						ok = true
						break
					}
				}

				if !ok {
					d.ftypes = append(d.ftypes, ft)
				}
			}

			d.ftypes = fts
		}
	}

	return f
}

// WithCallBack sets a callback function.
func WithCallBack(cb Opts) Opts {
	f := func(c Pipeline) {
		switch d := c.(type) {
		case *ChData:
			d.callback = cb
		case *VecData:
			d.callback = cb
		}
	}

	return f
}

// WithReader adds a reader.
func WithReader(rdr any) Opts {
	f := func(c Pipeline) {
		switch d := c.(type) {
		case *ChData:
			r, ok := rdr.(chutils.Input)
			if !ok {
				panic("reader not chutils.Input")
			}

			d.rdr = r
		}
	}

	return f
}

// SQLToPipe creates a pipe from the query sql
// Optional fts specifies the FTypes, usually to match an existing pipeline.
func SQLToPipe(sql string, fts FTypes, conn *chutils.Connect) (pipe Pipeline, err error) {
	rdr := s.NewReader(sql, conn)
	defer func() { _ = rdr.Close() }()

	if e := rdr.Init("", chutils.MergeTree); e != nil {
		return nil, e
	}

	pipe = NewChData("MSR Pipeline")

	if fts != nil {
		WithFtypes(fts)(pipe)
	}

	WithReader(rdr)(pipe)

	WithBatchSize(0)(pipe)
	if e := pipe.Init(); e != nil {
		return nil, e
	}

	return pipe, nil
}

// CSVToPipe creates a pipe from a CSV file
// Optional fts specifies the FTypes, usually to match an existing pipeline.
func CSVToPipe(csvFile string, fts FTypes) (pipe Pipeline, err error) {
	const tol = 0.98

	handle, ex := os.Open(csvFile)
	if ex != nil {
		return nil, ex
	}
	defer func() { _ = handle.Close() }()

	rdr := cf.NewReader(csvFile, ',', '\n', '"', 0, 1, 0, handle, 0)

	if e := rdr.Init("", chutils.MergeTree); e != nil {
		return nil, e
	}

	if e := rdr.TableSpec().Impute(rdr, 0, tol); e != nil {
		return nil, e
	}

	if e := rdr.Reset(); e != nil {
		return nil, e
	}

	pipe = NewChData("MSR Pipeline")

	if fts != nil {
		WithFtypes(fts)(pipe)
	}

	WithReader(rdr)(pipe)

	WithBatchSize(0)(pipe)
	if e := pipe.Init(); e != nil {
		return nil, e
	}

	return pipe, nil
}

// PipeToSQL creates "table" and saves the pipe data to it.
func PipeToSQL(pipe Pipeline, table string, conn *chutils.Connect) error {
	if table == "" {
		return fmt.Errorf("exportSQL: table cannot be empty")
	}

	// make writer
	wtr := s.NewWriter(table, conn)
	defer func() { _ = wtr.Close() }()

	gd := pipe.GData()
	tb := gd.TableSpec()

	if e := tb.Create(conn, table); e != nil {
		return e
	}

	if e := pipe.GData().Reset(); e != nil {
		return e
	}

	if e := chutils.Export(pipe.GData(), wtr, 0, false); e != nil {
		return e
	}

	return nil
}

// PipeToCSV saves the pipe as a CSV
func PipeToCSV(pipe Pipeline, outFile string) error {
	if outFile == "" {
		return fmt.Errorf("exportCSV: outFile cannot be empty")
	}

	handle, err := os.Create(outFile)
	if err != nil {
		return err
	}
	defer func() { _ = handle.Close() }()

	// write header
	if _, e := handle.WriteString(strings.Join(pipe.FieldList(), ",") + "\n"); e != nil {
		return e
	}

	// make writer
	wtr := cf.NewWriter(handle, "output", nil, ',', '\n', "tmp.xyz")
	defer func() { _ = wtr.Close() }()

	if e := pipe.GData().Reset(); e != nil {
		return e
	}

	// if after < 0, then won't also move to ClickHouse
	if e := chutils.Export(pipe.GData(), wtr, -1, false); e != nil {
		return e
	}

	return nil
}

// Join creates a new pipeline by joining pipe1 and pipe2 on joinField.
//   - JoinField must be categorical.
//   - The only field pipe1 and pipe2 can have in common is joinField
//   - pipe2 is sorted by Join.  Hence, pipe2 should be the smaller of the pipelines.
func Join(pipe1, pipe2 Pipeline, joinField string) (joined Pipeline, err error) {
	// fields in joined pipe
	field1, field2 := pipe1.FieldList(), pipe2.FieldList()

	// duplicate field names not allowed
	if e := disjoint(field1, field2, joinField); err != nil {
		return nil, e
	}

	// check join field is in both and is type FRCat
	if _, e := getOn(pipe1, joinField); e != nil {
		return nil, e
	}

	var on2 *GDatum
	if on2, err = getOn(pipe2, joinField); err != nil {
		return nil, err
	}

	// sort pipe2 by "joinField" field
	if e := pipe2.GData().Sort(joinField, true); e != nil {
		return nil, e
	}

	lvl2 := on2.FT.FP.Lvl
	on2Data := on2.Data.([]int32)

	n1, n2 := len(field1), len(field2)
	gd1, gd2 := pipe1.GData(), pipe2.GData()

	// The safest (though not fastest) way to do this is to recover the raw data from the pipelines
	raw1 := make([]*Raw, n1)
	indOn1 := -1 // index of joinField
	for cols := 0; cols < n1; cols++ {
		var e error
		raw1[cols], e = gd1.GetRaw(field1[cols])
		if e != nil {
			return nil, e
		}
		if field1[cols] == joinField {
			indOn1 = cols
		}
	}

	raw2 := make([]*Raw, n2)
	for cols := 0; cols < n2; cols++ {
		var e error
		raw2[cols], e = gd2.GetRaw(field2[cols])
		if e != nil {
			return nil, e
		}
	}

	joinRaw1 := make([][]any, n1)
	joinRaw2 := make([][]any, n2)

	for ind := 0; ind < pipe1.Rows(); ind++ {
		needle, ok := lvl2[raw1[indOn1].Data[ind]]
		if !ok {
			continue
		}

		loc2 := locInd(needle, on2Data)
		if loc2 < 0 {
			continue
		}

		// assemble columns
		for cols := 0; cols < n1; cols++ {
			joinRaw1[cols] = append(joinRaw1[cols], raw1[cols].Data[ind])
		}

		for cols := 0; cols < n2; cols++ {
			joinRaw2[cols] = append(joinRaw2[cols], raw2[cols].Data[loc2])
		}
	}

	gdata := &GData{}
	if e := addRaw(gdata, joinRaw1, field1, pipe1.GetFTypes(), ""); e != nil {
		return nil, e
	}
	if e := addRaw(gdata, joinRaw2, field2, pipe2.GetFTypes(), joinField); e != nil {
		return nil, e
	}

	joined = NewVecData("joined", gdata)

	return joined, nil
}

// addRaw adds fields to gdOutput.
// The fields added are specified by:
//   - inData - the actual data
//   - fieldNames - the names of the fields
//   - fts - the FType of the fields
//
// If joinField is in fieldNames, it is not added to gdOutput
func addRaw(gdOutput *GData, inData [][]any, fieldNames []string, fts FTypes, joinField string) error {
	for col := 0; col < len(fieldNames); col++ {
		rawcol := NewRaw(inData[col], nil)
		ft := fts[col]

		// on the join
		if fieldNames[col] == joinField {
			continue
		}

		switch ft.Role {
		case FRCat:
			if e := gdOutput.AppendD(rawcol, fieldNames[col], ft.FP); e != nil {
				return fmt.Errorf("addRaw error AppendD: %s", ft.Name)
			}
		default:
			if e := gdOutput.AppendC(rawcol, fieldNames[col], ft.Normalized, ft.FP); e != nil {
				return fmt.Errorf("addRaw error AppendC: %s", ft.Name)
			}
		}
	}

	return nil
}

// locInd finds the index of needle in haystack.  Return -1 if not there.
func locInd(needle int32, haystack []int32) int {
	ind := sort.Search(len(haystack), func(i int) bool { return haystack[i] >= needle })
	if haystack[ind] == needle {
		return ind
	}

	return -1
}

// getOn checks the joinField is present in the Pipeline
func getOn(pipe Pipeline, joinField string) (*GDatum, error) {
	gd := pipe.Get(joinField)

	if gd == nil {
		return nil, fmt.Errorf("join field %s not in pipe", joinField)
	}

	if gd.FT.Role != FRCat {
		return nil, fmt.Errorf("join field in pipe not type FRCat")
	}

	return gd, nil
}

// disjoint checks that fields1 and fields2 have no common elements aside from joinField
func disjoint(fields1, fields2 []string, joinField string) error {
	for _, fld1 := range fields1 {
		if fld1 == joinField {
			continue
		}

		for _, fld2 := range fields2 {
			if fld1 == fld2 {
				return fmt.Errorf("field %s in both pipelines", fld1)
			}
		}
	}

	return nil
}

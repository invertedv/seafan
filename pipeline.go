package seafan

// pipeline.go has the interface and "With" funcs for Pipelines.
import (
	"fmt"
	"os"
	"strings"

	"github.com/invertedv/utilities"

	"github.com/invertedv/chutils"
	cf "github.com/invertedv/chutils/file"
	s "github.com/invertedv/chutils/sql"
	G "gorgonia.org/gorgonia"
)

// The Pipeline interface specifies the methods required to be a data Pipeline. The Pipeline is the middleware between
// the data and the fitting routines.
type Pipeline interface {
	Init() error                                                              // initialize the pipeline
	Rows() int                                                                // # of observations in the pipeline (size of the epoch)
	Batch(inputs G.Nodes) bool                                                // puts the next batch in the input nodes
	Epoch(setTo int) int                                                      // manage epoch count
	IsNormalized(field string) bool                                           // true if feature is normalized
	IsCat(field string) bool                                                  // true if feature is one-hot encoded
	Cols(field string) int                                                    // # of columns in the feature
	IsCts(field string) bool                                                  // true if the feature is continuous
	GetFType(field string) *FType                                             // Get FType for the feature
	GetFTypes() FTypes                                                        // Get Ftypes for pipeline
	BatchSize() int                                                           // batch size
	FieldList() []string                                                      // fields available
	FieldCount() int                                                          // number of fields in the pipeline
	GData() *GData                                                            // return underlying GData
	Get(field string) *GDatum                                                 // return data for field
	GetKeepRaw() bool                                                         // returns whether raw data is kept
	Join(right Pipeline, onField string, joinType JoinType) (Pipeline, error) // joins two pipelines
	Slice(sl Slicer) (Pipeline, error)                                        // slice the pipeline
	Shuffle()                                                                 // shuffle data
	Describe(field string, topK int) string                                   // describes a field
	Subset(rows []int) (newPipe Pipeline, err error)                          // subsets pipeline to rows
	Where(field string, equalTo []any) (Pipeline, error)                      // subset pipeline to where field=equalTo
	Keep(fields []string) error                                               // keep on fields in the pipeline
	Drop(field string) error                                                  // drop field from the pipeline
	AppendRows(gd *GData, fTypes FTypes) (Pipeline, error)                    // appends gd to pipeline
	AppendRowsRaw(gd *GData) error                                            // appends gd ONLY to *Raw data
	ReInit(ftypes *FTypes) (Pipeline, error)                                  // reinitialized pipeline from *Raw data
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

// WithKeepRaw sets bool whether to keep the *Raw data in the pipeline.
func WithKeepRaw(keepRaw bool) Opts {
	f := func(c Pipeline) {
		switch d := c.(type) {
		case *ChData:
			d.keepRaw = keepRaw
		case *VecData:
			d.keepRaw = keepRaw
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
func SQLToPipe(sql string, fts FTypes, keepRaw bool, conn *chutils.Connect) (pipe Pipeline, err error) {
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
	WithKeepRaw(keepRaw)(pipe)

	WithBatchSize(0)(pipe)
	if e := pipe.Init(); e != nil {
		return nil, e
	}

	return pipe, nil
}

// CSVToPipe creates a pipe from a CSV file
// Optional fts specifies the FTypes, usually to match an existing pipeline.
func CSVToPipe(csvFile string, fts FTypes, keepRaw bool) (pipe Pipeline, err error) {
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
	WithKeepRaw(keepRaw)(pipe)

	if e := pipe.Init(); e != nil {
		return nil, e
	}

	return pipe, nil
}

// PipeToSQL creates "table" and saves the pipe data to it.
func PipeToSQL(pipe Pipeline, table string, after int, conn *chutils.Connect) error {
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

	if e := chutils.Export(pipe.GData(), wtr, after, false); e != nil {
		return e
	}

	return nil
}

// PipeToCSV saves the pipe as a CSV
func PipeToCSV(pipe Pipeline, outFile string, sep, eol, quote rune) error {
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
	wtr := cf.NewWriter(handle, "output", nil, sep, eol, quote, "tmp.xyz")
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

// Append appends pipe2 to the bottom of pipe1. pipe2 must have all the fields of pipe1 but may have extra,
// which are not in the returned pipe
func Append(pipe1, pipe2 Pipeline) (Pipeline, error) {
	if pipe1 == nil {
		return pipe2, nil
	}

	flds1 := pipe1.FieldList()
	flds2 := pipe2.FieldList()
	for _, fld := range flds1 {
		if utilities.Position(fld, "", flds2...) < 0 {
			return nil, fmt.Errorf("field %s not in append pipe", fld)
		}
	}

	gd1 := pipe1.GData()
	gd2 := pipe2.GData()
	forVec := make([][]any, gd1.FieldCount())
	for ind := 0; ind < len(flds1); ind++ {
		var (
			d1, d2 *Raw
			e      error
		)
		fld := flds1[ind]
		if d1, e = gd1.GetRaw(fld); e != nil {
			return nil, e
		}

		if d2, e = gd2.GetRaw(fld); e != nil {
			return nil, e
		}

		data := append(d1.Data, d2.Data...)
		forVec[ind] = data
	}

	return VecFromAny(forVec, flds1, nil)
}

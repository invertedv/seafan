package seafan

// pipeline.go has the interface and "With" funcs for Pipelines.
import (
	"github.com/invertedv/chutils"
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

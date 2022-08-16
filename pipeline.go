package seafan

import (
	"github.com/invertedv/chutils"
	G "gorgonia.org/gorgonia"
	"log"
)

// The Pipeline interface specifies the methods required to be a data pipeline
type Pipeline interface {
	Init() error                      // initialize the pipeline
	Rows() int                        // # of rows in the data set
	Batch(inputs G.Nodes) bool        // puts the next batch in the input nodes
	Epoch(setTo int) int              // manage epoch count
	IsNormalized(feature string) bool // true if feature is normalized
	IsCat(feature string) bool        // true if feature is one-hot encoded
	Cols(feature string) int
	//	Save
	IsCts(feature string) bool
	GetFeature(feature string) *FType
	BatchSize() int
	FieldList() []string
}

// Opts function sets an option to a Pipeline
type Opts func(c Pipeline)

// WithBatchSize sets the batch size for the pipeline
func WithBatchSize(bsize int) Opts {
	f := func(c Pipeline) {
		switch d := c.(type) {
		case *ChData:
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
				dt := d.Get(nm)
				if dt != nil {
					dt.FT.Role = FRCat
					continue
				}
				ft := &FType{
					Name: nm,
					Role: FRCat,
				}
				d.ftypes = append(d.ftypes, ft)
			}
		}
	}
	return f
}

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
		}
	}
	return f
}

// WithFParams sets the feature parameters: location/scale or map of feature value -> int32
func WithFtypes(fts FTypes) Opts {
	f := func(c Pipeline) {
		switch d := c.(type) {
		case *ChData:
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
		}
	}
	return f
}

// WithReader adds a reader. Supplying separately allows the user to work with those structures.
func WithReader(rdr any) Opts {
	f := func(c Pipeline) {
		switch d := c.(type) {
		case *ChData:
			r, ok := rdr.(chutils.Input)
			if !ok {
				log.Fatalln("reader not chutils.Input")
			}
			d.rdr = r
		}
	}
	return f
}

package seafan

// gdata.go implements structures and methods to produce gorgonia-ready data

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/stat"
)

type GDatum struct {
	FT      *FType  // FT stores the details of the field: it's role, # categories, mappings
	Summary Summary // Summary of the Data (e.g. distribution)
	Data    any     // Data. This will be either []float64 (FRCts, FROneHot, FREmbed) or []int32 (FRCat)
}

type GData []*GDatum

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
func (gd GData) check(name string) error {
	if name != "" {
		if gd.Get(name) != nil {
			return Wrapper(ErrGData, fmt.Sprintf("%s exists already", name))
		}
	}

	n := 0

	for _, d := range gd {
		if d.Summary.NRows != n && n > 0 {
			return Wrapper(ErrGData, "differing number of rows")
		}

		n = d.Summary.NRows
	}

	return nil
}

// AppendC appends a continuous feature
//
//goland:noinspection GoLinter,GoLinter,GoLinter,GoLinter,GoLinter,GoLinter
func (gd GData) AppendC(raw *Raw, name string, normalize bool, fp *FParam) (GData, error) {
	if e := gd.check(name); e != nil {
		return nil, e
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
				return nil, e
			}

			x[ind] = xx
		default:
			return nil, Wrapper(ErrGData, fmt.Sprintf("AppendC: cannot convert this type %T", x[0]))
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

	if ls.Scale < 1e-8 {
		return nil, Wrapper(ErrGData, fmt.Sprintf("AppendC: %s cannot be normalized--0 variance", name))
	}

	if normalize {
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
	gdOut := append(gd, c)

	if e := gdOut.check(""); e != nil {
		return nil, e
	}

	return gdOut, nil
}

// AppendD appends a discrete feature
func (gd GData) AppendD(raw *Raw, name string, fp *FParam) (GData, error) {
	if e := gd.check(name); e != nil {
		return nil, e
	}

	if fp == nil {
		lv := ByPtr(raw)
		fp = &FParam{Lvl: lv}
	}

	ds := make([]int32, len(raw.Data))

	for ind := 0; ind < len(ds); ind++ {
		v := raw.Data[ind]
		val, ok := fp.Lvl[v]

		if !ok {
			val, ok = fp.Lvl[fp.Default]
			if !ok {
				return nil, Wrapper(ErrGData, fmt.Sprintf("AppendD: default value %v not in dictionary", fp.Default))
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
	gdOut := append(gd, d)

	if e := gdOut.check(""); e != nil {
		return nil, e
	}

	return gdOut, nil
}

// MakeOneHot creates & appends a one hot feature from a discrete feature
func (gd GData) MakeOneHot(from, name string) (GData, error) {
	if e := gd.check(name); e != nil {
		return nil, e
	}

	d := gd.Get(from)

	if d == nil {
		return nil, Wrapper(ErrGData, fmt.Sprintf("MakeOneHot: 'from' feature %s not found", from))
	}

	if d.FT.Role != FRCat {
		return nil, Wrapper(ErrGData, fmt.Sprintf("MakeOneHot: input %s is not discrete", from))
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
	gdOut := append(gd, oH)

	if e := gdOut.check(""); e != nil {
		return nil, e
	}

	return gdOut, nil
}

// Get returns a single feature from GData
func (gd GData) Get(name string) *GDatum {
	for _, g := range gd {
		if g.FT.Name == name {
			return g
		}
	}

	return nil
}

// Slice creates a new GData sliced according to sl
func (gd GData) Slice(sl Slicer) (GData, error) {

	if sl == nil {
		return gd, nil
	}

	gOut := make(GData, 0)

	for _, g := range gd {
		ft := g.FT
		switch role := ft.Role; role {
		// These are all float64, but FROneHot and FREmbed are matrices
		case FRCts, FROneHot, FREmbed:
			cats := 1
			if role == FROneHot || role == FREmbed {
				cats = ft.Cats
			}
			d := make([]float64, 0)
			for row := 0; row < g.Summary.NRows; row++ {
				if sl(row) {
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
				NRows:  len(d),
				DistrC: desc,
				DistrD: nil,
			}
			datum := &GDatum{
				FT:      ftNew,
				Summary: summ,
				Data:    d,
			}
			gOut = append(gOut, datum)

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
			gOut = append(gOut, datum)
		}
	}

	return gOut, nil
}

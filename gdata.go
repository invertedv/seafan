package seafan

// structures and methods to produce gorgonia-ready data

import (
	"fmt"
	"gonum.org/v1/gonum/stat"
	"reflect"
	"strconv"
)

type GDatum struct {
	FT      *FType  // FT stores the details of the field: it's role, # categories, mappings
	Summary Summary // Summary of the Data (e.g. distribution)
	Data    any     // Data. This will be either []float64 (FRCts, FROneHot, FREmbed) or []int32 (FRCat)
}

type GData []*GDatum

// check performs a sanity check on GData
func (gd GData) check(name string) error {
	if name != "" {
		if gd.Get(name) != nil {
			return fmt.Errorf("%s exists already", name)
		}
	}
	n := 0
	for _, d := range gd {
		if d.Summary.nRow != n && n > 0 {
			return fmt.Errorf("differing number of rows")
		}
		n = d.Summary.nRow
	}
	return nil
}

// AppendC appends a continuous feature
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
			return nil, fmt.Errorf("cannot convert this type %T", x[0])
		}
	}
	ls := &FParam{}
	switch {
	case fp == nil:
		ls = &FParam{Location: stat.Mean(x, nil), Scale: stat.StdDev(x, nil)}
	case fp != nil:
		ls = fp
	}
	if ls.Scale < 1e-8 {
		return nil, fmt.Errorf("%s cannot be normalized--0 variance", name)
	}
	if normalize {
		for ind := 0; ind < len(x); ind++ {
			x[ind] = (x[ind] - ls.Location) / ls.Scale
		}
	}
	distr, _ := NewDesc(nil, name)
	distr.Populate(x, true)
	summ := Summary{
		nRow:   len(x),
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
				return nil, fmt.Errorf("default value %v not in dictionary", fp.Default)
			}
		}
		ds[ind] = val
	}
	distr := ByCounts(raw)
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
		nRow:   len(ds),
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
func (gd GData) MakeOneHot(from string, name string) (GData, error) {
	if e := gd.check(name); e != nil {
		return nil, e
	}
	d := gd.Get(from)
	if d == nil {
		return nil, fmt.Errorf("MakeOneHot: 'from' feature %s not found", from)
	}
	if d.FT.Role != FRCat {
		return nil, fmt.Errorf("MakeOneHot: input %s is not discrete", from)
	}
	nRow := d.Summary.nRow
	nCat := len(d.FT.FP.Lvl)
	oh := make([]float64, nRow*nCat)
	for row := 0; row < nRow; row++ {
		oh[int32(row*nCat)+d.Data.([]int32)[row]] = 1
	}
	summ := Summary{nRow: d.Summary.nRow}
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

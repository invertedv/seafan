package seafan

// data.go has structure and methods for basic data types in seafan.

import (
	"fmt"
	"math"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"

	grob "github.com/MetalBlueberry/go-plotly/graph_objects"
	"gonum.org/v1/gonum/stat"
)

// XY struct holds (x,y) pairs as distinct slices
type XY struct {
	X []float64
	Y []float64
}

// NewXY creates a pointer to a new XY with error checking
func NewXY(x, y []float64) (*XY, error) {
	if x == nil && y == nil {
		return &XY{}, nil
	}

	if (x == nil && y != nil) || (x != nil && y == nil) {
		return nil, Wrapper(ErrData, "NewXY: one of x,y is nil and one is not")
	}

	if len(x) != len(y) {
		return nil, Wrapper(ErrData, fmt.Sprintf("NewXY: x has length %d and y has different length of %d", len(x), len(y)))
	}

	return &XY{X: x, Y: y}, nil
}

func (p *XY) Swap(i, j int) {
	p.X[i], p.X[j] = p.X[j], p.X[i]
	p.Y[i], p.Y[j] = p.Y[j], p.Y[i]
}

func (p *XY) Less(i, j int) bool {
	return p.X[i] < p.X[j]
}

func (p *XY) Len() int {
	return len(p.X)
}

// Sort sorts with error checking
func (p *XY) Sort() error {
	if len(p.X) != len(p.Y) {
		return Wrapper(ErrData, "(*XY).Sort: X and Y must have same length")
	}

	sort.Sort(p)

	return nil
}

// Interp linearly interpolates XY at the points xNew.
func (p *XY) Interp(xNew []float64) (*XY, error) {
	if len(p.X) != len(p.Y) {
		return nil, Wrapper(ErrData, "(*XY).Interp: X and Y must have same length")
	}

	if !sort.Float64sAreSorted(p.X) {
		sort.Sort(p)
	}

	yNew := make([]float64, len(xNew))

	for ind, xn := range xNew {
		i := sort.SearchFloat64s(p.X, xn)

		switch {
		case i == len(p.X):
			yNew[ind] = p.Y[i-1]
		case p.X[i] == xn:
			yNew[ind] = p.Y[i]
		case i == 0:
			yNew[ind] = p.Y[0]
		default:
			w := (xn - p.X[i-1]) / (p.X[i] - p.X[i-1])
			yNew[ind] = w*p.Y[i] + (1.0-w)*p.Y[i-1]
		}
	}

	return &XY{X: xNew, Y: yNew}, nil
}

func (p *XY) String() string {
	s := "     X                 Y\n"
	for ind := 0; ind < len(p.X); ind++ {
		s = fmt.Sprintf("%s%10f        %10f\n", s, p.X[ind], p.Y[ind])
	}

	return s
}

// Plot produces an XY Plotly plot
func (p *XY) Plot(pd *PlotDef, scatter bool) error {
	if len(p.X) != len(p.Y) {
		return Wrapper(ErrData, "(*XY).Plot: X and Y must have same length")
	}

	var sType grob.ScatterMode

	switch scatter {
	case true:
		sType = grob.ScatterModeMarkers
	default:
		sType = grob.ScatterModeLines
	}

	tr := &grob.Scatter{
		Type: grob.TraceTypeScatter,
		X:    p.X,
		Y:    p.Y,
		Name: "Scatter",
		Mode: sType,
		Line: &grob.ScatterLine{Color: "black"},
	}

	fig := &grob.Fig{Data: grob.Traces{tr}}

	return Plotter(fig, nil, pd)
}

// Desc contains descriptive information of a float64 slice
type Desc struct {
	Name string    // Name is the name of feature we are describing
	N    int       // N is the number of observations
	U    []float64 // U is the slice of locations at which to find the quantile
	Q    []float64 // Q is the slice of empirical quantiles
	Mean float64   // Mean is the average of the data
	Std  float64   // standard deviation
}

// NewDesc creates a pointer to a new Desc struct instance with error checking.
//
//	u is a slice of values at which to find quantiles. If nil, a standard set is used.
//	name is the name of the feature (for printing)(
func NewDesc(u []float64, name string) (*Desc, error) {
	if u == nil {
		u = []float64{0, .1, .25, .5, .75, .9, 1.0}
	}

	sort.Float64s(u)

	if u[0] < 0.0 || u[len(u)-1] > 1.0 {
		return nil, Wrapper(ErrData, "NewDesc: quantiles best be in [0,1]")
	}

	d := &Desc{U: u, Name: name}
	d.Q = make([]float64, len(u))

	return d, nil
}

// Populate calculates the descriptive statistics based on x.
// The slice is not sorted if noSort
func (d *Desc) Populate(x []float64, noSort bool, sl Slicer) {
	// see if we need to sort this
	xIn := x

	if sl != nil {
		noSort = false
		xIn = make([]float64, 0)
		for row := 0; row < len(x); row++ {
			if sl(row) {
				xIn = append(xIn, x[row])
			}
		}
	}

	if !sort.Float64sAreSorted(xIn) {
		switch noSort {
		case false:
			sort.Float64s(xIn)
		case true:
			xIn = make([]float64, len(x))
			copy(xIn, x)
			sort.Float64s(xIn)
		}
	}

	for ind, u := range d.U {
		d.Q[ind] = stat.Quantile(u, stat.Empirical, xIn, nil)
	}

	d.N = len(xIn)
	d.Mean, d.Std = stat.MeanStdDev(xIn, nil)
}

func (d *Desc) String() string {
	s := fmt.Sprintf("Descriptive Statistics for %s\n", d.Name)
	s = fmt.Sprintf("%sn               %d\n", s, d.N)
	s = fmt.Sprintf("%sMean            %f\n", s, d.Mean)
	s = fmt.Sprintf("%sStd Dev         %f\n", s, d.Std)

	for ind := 0; ind < len(d.U); ind++ {
		s = fmt.Sprintf("%sQ(%0.2f)         %f\n", s, d.U[ind], d.Q[ind])
	}

	return s
}

// Raw holds a raw slice of type Kind
type Raw struct {
	Kind reflect.Kind // type of elements of Data
	Data []any
}

// NewRaw creates a new raw slice from x.  This assumes all elements of x are the same Kind
func NewRaw(x []any, sl Slicer) *Raw {
	if x == nil {
		return nil
	}

	if sl != nil {
		xSlice := make([]any, 0)

		for row := 0; row < len(x); row++ {
			if sl(row) {
				xSlice = append(xSlice, x[row])
			}
		}

		return &Raw{Data: xSlice, Kind: reflect.TypeOf(xSlice[0]).Kind()}
	}

	return &Raw{Data: x, Kind: reflect.TypeOf(x[0]).Kind()}
}

func NewRawCast(x any, sl Slicer) *Raw {
	xOut := make([]any, 0)
	switch d := x.(type) {
	case []string:
		for row := 0; row < len(d); row++ {
			switch sl == nil {
			case true:
				xOut = append(xOut, d[row])
			case false:
				if sl(row) {
					xOut = append(xOut, d[row])
				}
			}
		}
	case []int32:
		for row := 0; row < len(d); row++ {
			switch sl == nil {
			case true:
				xOut = append(xOut, d[row])
			case false:
				if sl(row) {
					xOut = append(xOut, d[row])
				}
			}
		}
	case []int64:
		for row := 0; row < len(d); row++ {
			switch sl == nil {
			case true:
				xOut = append(xOut, d[row])
			case false:
				if sl(row) {
					xOut = append(xOut, d[row])
				}
			}
		}
	case []float64:
		for row := 0; row < len(d); row++ {
			switch sl == nil {
			case true:
				xOut = append(xOut, d[row])
			case false:
				if sl(row) {
					xOut = append(xOut, d[row])
				}
			}
		}
	case []float32:
		for row := 0; row < len(d); row++ {
			switch sl == nil {
			case true:
				xOut = append(xOut, d[row])
			case false:
				if sl(row) {
					xOut = append(xOut, d[row])
				}
			}
		}
	default:
		return nil
	}
	return NewRaw(xOut, nil)
}

// AllocRaw creates an empty slice of type kind and len n
func AllocRaw(n int, kind reflect.Kind) *Raw {
	return &Raw{Data: make([]any, n), Kind: kind}
}

// GTAny compares xa > xb
func GTAny(xa, xb any) (truth bool, err error) {
	if xb == nil || xa == nil {
		return true, nil
	}

	if reflect.TypeOf(xa) != reflect.TypeOf(xb) {
		return false, fmt.Errorf("must be of same type")
	}

	switch x := xa.(type) {
	case string:
		x = strings.ReplaceAll(x, "'", "")
		y := strings.ReplaceAll(xb.(string), "'", "")
		return x > y, nil
	case int32:
		return x > xb.(int32), nil
	case int64:
		return x > xb.(int64), nil
	case float32:
		return x > xb.(float32), nil
	case float64:
		return x > xb.(float64), nil
	case time.Time:
		tmp := x.Sub(xb.(time.Time))
		_ = tmp
		return x.Sub(xb.(time.Time)) > 0, nil
	}

	return false, fmt.Errorf("unsupported comparison")
}

// Sum sums elements
func (r *Raw) Sum() (*Raw, error) {
	if r.Data == nil {
		return nil, fmt.Errorf("no data: (*Raw) Sum")
	}

	if !r.IsNumeric() {
		return nil, fmt.Errorf("coversion to float64 not possible for %v, (*Raw) Product", r.Kind)
	}

	s := 0.0
	for _, val := range r.Data {
		x := Any2Float64(val)
		if x == nil {
			return nil, fmt.Errorf("conversion to float64 error (*Raw) Sum")
		}
		s += x.(float64)
	}

	return NewRaw([]any{s}, nil), nil
}

// Product returns the product of the elements
func (r *Raw) Product() (*Raw, error) {
	if r.Data == nil {
		return nil, fmt.Errorf("no data: (*Raw) Sum")
	}

	if !r.IsNumeric() {
		return nil, fmt.Errorf("coversion to float64 not possible for %v, (*Raw) Product", r.Kind)
	}

	s := 1.0
	for _, val := range r.Data {
		x := Any2Float64(val)
		if x == nil {
			return nil, fmt.Errorf("conversion to float64 error (*Raw) Product")
		}
		s *= x.(float64)
	}

	return NewRaw([]any{s}, nil), nil
}

// Mean finds the average
func (r *Raw) Mean() (*Raw, error) {
	meanR, e := r.Sum()
	if e != nil {
		return nil, e
	}

	mean := meanR.Data[0].(float64) / float64(r.Len())

	return NewRaw([]any{mean}, nil), nil
}

// Std finds the sample standard deviation
func (r *Raw) Std() (*Raw, error) {
	meanR, e := r.Mean()
	if e != nil {
		return nil, e
	}

	if r.Len() == 1 {
		return NewRaw([]any{0.0}, nil), nil
	}

	std := 0.0
	nFlt := float64(r.Len())
	mean := meanR.Data[0].(float64)

	for _, valx := range r.Data {
		delta := Any2Float64(valx).(float64) - mean
		std += delta * delta
	}

	std = math.Sqrt(std / (nFlt - 1))

	return NewRaw([]any{std}, nil), nil
}

// Max returns max
func (r *Raw) Max() (*Raw, error) {
	var s any
	for _, val := range r.Data {
		gt, e := GTAny(val, s)
		if e != nil {
			return nil, e
		}

		if gt {
			s = val
		}
	}

	return NewRaw([]any{s}, nil), nil
}

// Min returns min
func (r *Raw) Min() (*Raw, error) {
	s := r.Data[0]
	for _, val := range r.Data {
		gt, e := GTAny(val, s)
		if e != nil {
			return nil, e
		}

		if !gt {
			s = val
		}
	}

	return NewRaw([]any{s}, nil), nil
}

// IsNumeric returns true if the underlying type is numeric
func (r *Raw) IsNumeric() bool {
	if r.Data == nil {
		return false
	}

	switch r.Kind {
	case reflect.Float32, reflect.Float64, reflect.Int, reflect.Int32, reflect.Int64:
		return true
	default:
		return false
	}
}

// CumeAfter cumulates the data after the current row, for each row.
//
//	AggType can take on the following values:
//	- "sum"  Cumulative sums are taken.
//	- "product" Cumulative products are taken.
//	- "count" Counts for rows are taken.
//
// For "sum" and "product", the value "missing" is used for the last row.
func (r *Raw) CumeAfter(missing any, aggType string) (*Raw, error) {
	if !r.IsNumeric() {
		return nil, fmt.Errorf("numeric operation on %v", r.Kind)
	}

	// coerce to same type as r
	var miss any
	if missing != nil {
		miss = Any2Kind(missing, r.Kind)
		if miss == nil {
			return nil, fmt.Errorf("cannot convert %v to %v (*Raw) Lag", missing, r.Kind)
		}
	}

	cumes := make([]any, r.Len())

	for ind := 0; ind < r.Len(); ind++ {
		if ind < r.Len()-1 {
			var e error
			var result any
			var data *Raw
			switch aggType {
			case "sum":
				data, e = NewRaw(r.Data[ind+1:], nil).Sum()
				result = data.Data[0]
			case "product":
				data, e = NewRaw(r.Data[ind+1:], nil).Product()
				result = data.Data[0]
			case "count":
				result = any(float64(r.Len() - 1 - ind))
			default:
				return nil, fmt.Errorf("unknown aggType (*Raw) CumeAfter")
			}

			if e != nil {
				return nil, e
			}

			cumes[ind] = result
		} else {
			switch aggType {
			case "sum", "product":
				cumes[ind] = miss
			default:
				cumes[ind] = float64(ind - 1)
			}
		}
	}

	return NewRaw(cumes, nil), nil
}

// CumeBefore cumulates the data before the current row, for each row.
//
//	AggType can take on the following values:
//	- "sum"  Cumulative sums are taken.
//	- "product" Cumulative products are taken.
//	- "count", "row" Counts for rows are taken.
//
// For "sum" and "product", the value "missing" is used for the first row.
func (r *Raw) CumeBefore(missing any, aggType string) (*Raw, error) {
	if !r.IsNumeric() {
		return nil, fmt.Errorf("numeric operation on %v", r.Kind)
	}

	// coerce to same type as r
	var miss any
	if missing != nil {
		miss = Any2Kind(missing, r.Kind)
		if miss == nil {
			return nil, fmt.Errorf("cannot convert %v to %v (*Raw) Lag", missing, r.Kind)
		}
	}

	cumes := make([]any, r.Len())

	for ind := 0; ind < r.Len(); ind++ {
		if ind > 0 {
			var data *Raw
			var e error
			var result any
			switch aggType {
			case "sum":
				data, e = NewRaw(r.Data[:ind], nil).Sum()
				result = data.Data[0]
			case "product":
				data, e = NewRaw(r.Data[:ind], nil).Product()
				result = data.Data[0]
			case "count", "row":
				result = float64(ind)
			}
			if e != nil {
				return nil, e
			}
			cumes[ind] = result
		} else {
			switch aggType {
			case "sum", "product":
				cumes[ind] = miss
			default:
				cumes[ind] = float64(ind)
			}
		}
	}

	return NewRaw(cumes, nil), nil
}

// Lag returns r lagged by 1.  The first element is set to "missing".
func (r *Raw) Lag(missing any) (*Raw, error) {
	if r.Data == nil {
		return nil, fmt.Errorf("no data: (*Raw) Lag")
	}

	// coerce to same type as r
	miss := Any2Kind(missing, r.Kind)
	if miss == nil {
		return nil, fmt.Errorf("cannot convert %v to %v (*Raw) Lag", missing, r.Kind)
	}

	xOut := make([]any, r.Len())
	xOut[0] = miss

	for ind := 1; ind < r.Len(); ind++ {
		xOut[ind] = r.Data[ind-1]
	}

	return NewRaw(xOut, nil), nil
}

// Log takes the natural log of Raw
func (r *Raw) Log() (*Raw, error) {
	if !r.IsNumeric() {
		return nil, fmt.Errorf("numeric operation on %v", r.Kind)
	}

	xOut := make([]any, r.Len())
	for ind, xval := range r.Data {
		x := Any2Float64(xval).(float64)
		if x <= 0 {
			return nil, fmt.Errorf("log of non-positive number (*Raw) Log: %v", x)
		}
		xOut[ind] = math.Log(x)
	}

	return NewRaw(xOut, nil), nil
}

// Exp returns e to the Raw
func (r *Raw) Exp() (*Raw, error) {
	if !r.IsNumeric() {
		return nil, fmt.Errorf("numeric operation on %v", r.Kind)
	}

	xOut := make([]any, r.Len())
	for ind, xval := range r.Data {
		x := Any2Float64(xval).(float64)
		xOut[ind] = math.Exp(x)
	}

	return NewRaw(xOut, nil), nil
}

// Pow returns Raw^exponent
func (r *Raw) Pow(exponent *Raw) (*Raw, error) {
	if !r.IsNumeric() || !exponent.IsNumeric() {
		return nil, fmt.Errorf("numeric operation on %v %v (*Raw) Pow", r.Kind, exponent.Kind)
	}

	delta1, delta2 := 1, 1
	if r.Len() == 1 {
		delta1 = 0
	}
	if exponent.Len() == 1 {
		delta2 = 0
	}

	if delta1 == 1 && delta2 == 1 && r.Len() != exponent.Len() {
		return nil, fmt.Errorf("exponent and base must have the same length, if not length=1 (*Raw) Pow")
	}

	n := r.Len()
	if m := exponent.Len(); m > n {
		n = m
	}

	xOut := make([]any, r.Len())
	ind1, ind2 := 0, 0
	for ind := 0; ind < n; ind++ {
		base := Any2Float64(r.Data[ind1]).(float64)
		exp := Any2Float64(exponent.Data[ind2]).(float64)
		xOut[ind] = math.Pow(base, exp)
	}

	return NewRaw(xOut, nil), nil
}

// Index returns data that is *Raw at the indices "indices"
func (r *Raw) Index(indices *Raw) (*Raw, error) {
	if !indices.IsNumeric() {
		return nil, fmt.Errorf("indices must be numeric (*Raw) Index")
	}

	xOut := make([]any, indices.Len())
	for ind, indval := range indices.Data {
		index := Any2Int(indval).(int)
		if index < 0 || index >= r.Len() {
			return nil, fmt.Errorf("index out of range: %d (*Raw) Index", index)
		}
		xOut[ind] = r.Data[index]
	}

	return NewRaw(xOut, nil), nil
}

func (r *Raw) Less(i, j int) bool {
	v, err := AnyLess(r.Data[i], r.Data[j])

	if err != nil {
		return false
	}

	return v
}

func (r *Raw) Swap(i, j int) {
	r.Data[i], r.Data[j] = r.Data[j], r.Data[i]
}

func (r *Raw) Len() int {
	return len(r.Data)
}

// Levels is a map from underlying values if a discrete tensor to int32 values
type Levels map[any]int32

// ByCounts builds a Levels map with the distribution of data
func ByCounts(data *Raw, sl Slicer) Levels {
	l := make(Levels)
	for row := 0; row < data.Len(); row++ {
		v := data.Data[row]
		switch sl == nil {
		case true:
			l[v]++
		case false:
			if sl(row) {
				l[v]++
			}
		}
	}

	return l
}

// ByPtr returns a mapping of values of data to []int32 for modeling.  The values of data are sorted, so the
// smallest will have a mapped value of 0.
func ByPtr(data *Raw) Levels {
	us := Unique(data.Data)
	bm := NewRaw(us, nil)
	sort.Sort(bm)

	l := make(Levels)

	for ind := 0; ind < len(bm.Data); ind++ {
		l[bm.Data[ind]] = int32(ind)
	}

	return l
}

// kv is used to sort maps either by the key or value
type kv struct {
	ord    []int
	kv     []any
	ascend bool
}

func (x *kv) Less(i, j int) bool {
	v, err := AnyLess(x.kv[i], x.kv[j])
	if err != nil {
		return false
	}
	if !x.ascend {
		v = !v
	}

	return v
}

func (x *kv) Swap(i, j int) {
	x.ord[i], x.ord[j] = x.ord[j], x.ord[i]
	x.kv[i], x.kv[j] = x.kv[j], x.kv[i]
}

func (x *kv) Len() int {
	return len(x.ord)
}

// pad returns a padding string of length maxLen-thisLen
func pad(maxLen, thisLen int) string {
	sp := "   "
	for ind := 0; ind < maxLen-thisLen; ind++ {
		sp += " "
	}

	return sp
}

// FindValue returns key that maps to val
func (l Levels) FindValue(val int32) any {
	for k, v := range l {
		if v == val {
			return k
		}
	}
	return nil
}

// Sort sorts Levels, returns sorted map as key, val slices
func (l Levels) Sort(byName, ascend bool) (key []any, val []int32) {
	inKey := make([]any, len(l))
	inVal := make([]any, len(l))
	ord := make([]int, len(l))
	ind := 0

	for kx, v := range l {
		inKey[ind] = kx
		inVal[ind] = v
		ord[ind] = ind
		ind++
	}
	key = make([]any, 0)
	val = make([]int32, 0)
	switch byName {
	case true:
		kvx := &kv{ord: ord, kv: inKey, ascend: ascend}
		sort.Sort(kvx)
		for indx := 0; indx < len(inKey); indx++ {
			key = append(key, inKey[indx])
			val = append(val, inVal[ord[indx]].(int32))
		}

	case false:
		kvx := &kv{ord: ord, kv: inVal, ascend: ascend}
		sort.Sort(kvx)
		for indx := 0; indx < len(inKey); indx++ {
			key = append(key, inKey[ord[indx]])
			val = append(val, inVal[indx].(int32))
		}
	}
	return key, val
}

// TopK returns the top k values either by name or by counts, ascending or descending
func (l Levels) TopK(topNum int, byName, ascend bool) string {
	keyS, valS := l.Sort(byName, ascend)

	maxLen := 11 // "Field Value" length

	for kx := range l {
		maxLen = Max(maxLen, len(fmt.Sprintf("%v", kx)))
	}

	if topNum <= 0 {
		topNum = len(keyS)
	}

	str := fmt.Sprintf("Field Value%sCount\n", pad(maxLen, 11))
	for ind := 0; ind < Min(topNum, len(keyS)); ind++ {
		keyS := fmt.Sprintf("%v", keyS[ind])
		str = fmt.Sprintf("%s%s%s%v\n", str, keyS, pad(maxLen, len(keyS)), valS[ind])
	}

	return str
}

// AnyLess returns x<y for select underlying types of "any"
func AnyLess(x, y any) (bool, error) {
	switch xt := x.(type) {
	case float64:
		return xt < y.(float64), nil
	case float32:
		return xt < y.(float32), nil
	case int64:
		return xt < y.(int64), nil
	case int32:
		return xt < y.(int32), nil
	case string:
		return xt < y.(string), nil
	case time.Time:
		return y.(time.Time).Sub(xt) > 0, nil
	default:
		return false, Wrapper(ErrData, "AnyLess: no comparison")
	}
}

// Min returns the Min of a & b
func Min(a, b int) int {
	if a < b {
		return a
	}

	return b
}

// Max returns the Max of a & b
func Max(a, b int) int {
	if a > b {
		return a
	}

	return b
}

// Unique returns a slice of the unique values of xs
func Unique(xs []any) []any {
	u := make([]any, 0)
	l := make(map[any]int)

	for _, x := range xs {
		if _, ok := l[x]; !ok {
			l[x] = 1

			u = append(u, x)
		}
	}

	return u
}

// Comparer compares xa and xb
func Comparer(xa, xb any, comp string) (truth bool, err error) {
	// a constant date comes in as a string
	if t1 := Any2Date(xa); t1 != nil {
		xa = t1
	}

	if t2 := Any2Date(xb); t2 != nil {
		xb = t2
	}

	test1, e1 := GTAny(xa, xb)
	if e1 != nil {
		return false, e1
	}

	test2, e2 := GTAny(xb, xa)
	if e2 != nil {
		return false, e2
	}

	switch comp {
	case ">":
		return test1, nil
	case ">=":
		return !test2, nil
	case "==":
		return !test1 && !test2, nil
	case "!=":
		return test1 || test2, nil
	case "<":
		return test2, nil
	case "<=":
		return !test1, nil
	}

	return false, fmt.Errorf("unsupported comparison: %s", comp)
}

// Any2Date attempts to convert inVal to a date (time.Time)
func Any2Date(inVal any) any {
	switch x := inVal.(type) {
	case string:
		formats := []string{"20060102", "1/2/2006", "01/02/2006"}
		for _, fmtx := range formats {
			t1, e := time.Parse(fmtx, strings.ReplaceAll(x, "'", ""))
			if e == nil {
				return t1
			}
		}
	case time.Time:
		return x
	}

	return nil
}

// Any2Float64 attempts to convert inVal to float64
func Any2Float64(inVal any) any {
	switch x := inVal.(type) {
	case int:
		return float64(x)
	case int32:
		return float64(x)
	case int64:
		return float64(x)
	case float32:
		return float64(x)
	case float64:
		return x
	case string:
		xx, e := strconv.ParseFloat(x, 64)
		if e != nil {
			return nil
		}
		return xx
	default:
		return nil
	}
}

// Any2Float32 attempts to convert inVal to float32
func Any2Float32(inVal any) any {
	switch x := inVal.(type) {
	case int:
		return float32(x)
	case int32:
		return float32(x)
	case int64:
		return float32(x)
	case float32:
		return x
	case float64:
		return x
	case string:
		xx, e := strconv.ParseFloat(x, 32)
		if e != nil {
			return nil
		}
		return float32(xx)
	default:
		return nil
	}
}

// Any2Int64 attempts to convert inVal to int64
func Any2Int64(inVal any) any {
	switch x := inVal.(type) {
	case int:
		return int64(x)
	case int32:
		return int64(x)
	case int64:
		return x
	case float32:
		return int64(x)
	case float64:
		return int64(x)
	case string:
		xx, e := strconv.ParseInt(x, 10, 64)
		if e != nil {
			return nil
		}
		return xx
	default:
		return nil
	}
}

// Any2Int32 attempts to convert inVal to int32
func Any2Int32(inVal any) any {
	switch x := inVal.(type) {
	case int:
		return int32(x)
	case int32:
		return x
	case int64:
		return int32(x)
	case float32:
		return int32(x)
	case float64:
		return int32(x)
	case string:
		xx, e := strconv.ParseInt(x, 10, 32)
		if e != nil {
			return nil
		}
		return int32(xx)
	default:
		return nil
	}
}

// Any2Int attempts to convert inVal to int
func Any2Int(inVal any) any {
	switch x := inVal.(type) {
	case int:
		return x
	case int32:
		return int(x)
	case int64:
		return int(x)
	case float32:
		return int(x)
	case float64:
		return int(x)
	case string:
		xx, e := strconv.ParseInt(x, 10, 32)
		if e != nil {
			return nil
		}
		return int(xx)
	default:
		return nil
	}
}

func Any2String(inVal any) any {
	switch x := inVal.(type) {
	case string:
		return x
	case time.Time:
		return x.Format("1/2/2006")
	case float32, float64:
		return fmt.Sprintf("%0.2f", x)
	default:
		return fmt.Sprintf("%v", x)
	}
}

func Any2Kind(inVal any, kind reflect.Kind) any {
	if inVal == nil {
		return nil
	}

	switch kind {
	case reflect.Float64:
		return Any2Float64(inVal)
	case reflect.Float32:
		return Any2Float32(inVal)
	case reflect.Int64:
		return Any2Int64(inVal)
	case reflect.Int32:
		return Any2Int32(inVal)
	case reflect.Int:
		return Any2Int(inVal)
	case reflect.String:
		return Any2String(inVal)
	case reflect.Struct:
		return Any2Date(inVal)
	default:
		return nil
	}
}

// str2Kind converts a string specifying a type to the reflect.Kind
func str2Kind(str string) reflect.Kind {
	switch str {
	case "float64":
		return reflect.Float64
	case "float32":
		return reflect.Float32
	case "string":
		return reflect.String
	case "int":
		return reflect.Int
	case "int32":
		return reflect.Int32
	case "int64":
		return reflect.Int64
	case "time.Time":
		return reflect.Struct
	default:
		return reflect.Interface
	}
}

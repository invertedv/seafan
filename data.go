package seafan

// data.go has structure and methods for basic data types in seafan.

import (
	"fmt"
	grob "github.com/MetalBlueberry/go-plotly/graph_objects"
	"gonum.org/v1/gonum/stat"
	"reflect"
	"sort"
)

// XY struct holds (x,y) pairs as distinct slices
type XY struct {
	X []float64
	Y []float64
}

// NewXY creates a pointer to a new XY with error checking
func NewXY(x []float64, y []float64) (*XY, error) {
	if x == nil && y == nil {
		return &XY{}, nil
	}
	if (x == nil && y != nil) || (x != nil && y == nil) {
		return nil, fmt.Errorf("one of x,y is nil and one not")
	}
	if len(x) != len(y) {
		return nil, fmt.Errorf("x has length %d and y has differnet length of %d", len(x), len(y))
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
		return fmt.Errorf("X and Y must have same length")
	}
	sort.Sort(p)
	return nil
}

// Interp linearly interpolates XY at the points xNew.
func (p *XY) Interp(xNew []float64) (*XY, error) {
	if len(p.X) != len(p.Y) {
		return nil, fmt.Errorf("X and Y must have same length")
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
		return fmt.Errorf("X and Y must have same length")
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
		return nil, fmt.Errorf("quantiles best be in [0,1]")
	}
	d := &Desc{U: u, Name: name}
	d.Q = make([]float64, len(u))
	return d, nil
}

// Populate calculates the descriptive statistics based on x.
// The slice is not sorted if noSort
func (d *Desc) Populate(x []float64, noSort bool) {
	// see if we need to sort this
	xIn := x
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
	d.N = len(x)
	d.Mean = stat.Mean(x, nil)
	d.Std = stat.StdDev(x, nil)
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
func NewRaw(x []any) *Raw {
	if x == nil {
		return nil
	}
	return &Raw{Data: x, Kind: reflect.TypeOf(x[0]).Kind()}
}

// AllocRaw creates an empty slice of type kind and len n
func AllocRaw(n int, kind reflect.Kind) *Raw {
	return &Raw{Data: make([]any, n), Kind: kind}
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
func ByCounts(data *Raw) Levels {
	l := make(Levels)
	for _, v := range data.Data {
		l[v]++
	}
	return l
}

// ByPtr returns a mapping of values of xs to []int32 for modeling.  The values of xs are sorted, so the
// smallest will have a mapped value of 0.
func ByPtr(data *Raw) Levels {
	us := Unique(data.Data)
	bm := NewRaw(us)
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

// TopK returns the top k values either by name or by counts, ascending or descending
func (l Levels) TopK(k int, byName bool, ascend bool) string {
	key := make([]any, len(l))
	val := make([]any, len(l))
	ord := make([]int, len(l))
	ind := 0
	maxLen := 11 // "Field Value" length
	for kx, v := range l {
		key[ind] = kx
		val[ind] = v
		ord[ind] = ind
		maxLen = Max(maxLen, len(fmt.Sprintf("%v", kx)))
		ind++
	}
	if k == 0 {
		k = len(key)
	}
	switch byName {
	case true:
		kvx := &kv{ord: ord, kv: key, ascend: ascend}
		sort.Sort(kvx)
		str := fmt.Sprintf("Field Value%sCount\n", pad(maxLen, 11))
		for ind := 0; ind < Min(k, len(key)); ind++ {
			keyS := fmt.Sprintf("%v", key[ind])
			str = fmt.Sprintf("%s%s%s%v\n", str, keyS, pad(maxLen, len(keyS)), val[ord[ind]])
		}
		return str
	case false:
		kvx := &kv{ord: ord, kv: val, ascend: ascend}
		sort.Sort(kvx)
		str := fmt.Sprintf("Field Value%sCount\n", pad(maxLen, 11))
		for ind := 0; ind < Min(k, len(key)); ind++ {
			keyS := fmt.Sprintf("%v", key[ord[ind]])
			str = fmt.Sprintf("%s%s%s%v\n", str, keyS, pad(maxLen, len(keyS)), val[ind])
		}
		return str
	}
	return ""
}

// AnyLess returns x<y for select underlying types of "any"
func AnyLess(x, y any) (bool, error) {
	switch x.(type) {
	case float64:
		return x.(float64) < y.(float64), nil
	case float32:
		return x.(float32) < y.(float32), nil
	case int64:
		return x.(int64) < y.(int64), nil
	case int32:
		return x.(int32) < y.(int32), nil
	case string:
		return x.(string) < y.(string), nil
	default:
		return false, fmt.Errorf("no comparison")
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
func Unique(xs []any) (u []any) {
	u = make([]any, 0)
	l := make(map[any]int)
	for _, x := range xs {
		if _, ok := l[x]; !ok {
			l[x] = 1
			u = append(u, x)
		}
	}
	return u
}

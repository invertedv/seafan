package seafan

import (
	"math/rand"
	"reflect"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestXY_Interp(t *testing.T) {
	x := []float64{3.0, 1.0, 2.0, 5.0}
	y := []float64{1.0, 2.0, 3.0, 4.0}
	xy, e := NewXY(x, y)
	assert.Nil(t, e)
	xNew := []float64{1.5, 2.2}
	expectY := []float64{2.5, 3 - .2*2}
	xyi, e := xy.Interp(xNew)
	assert.Nil(t, e)
	for ind := 0; ind < len(expectY); ind++ {
		assert.InEpsilon(t, xyi.Y[ind], expectY[ind], .000001)
	}
}

func TestXY_Sort(t *testing.T) {
	x := []float64{3, 1, 2, 5}
	y := []float64{1, 2, 3, 4}
	expectX := []float64{1, 2, 3, 5}
	expectY := []float64{2, 3, 1, 4}
	xy, e := NewXY(x, y)
	assert.Nil(t, e)
	e = xy.Sort()
	assert.Nil(t, e)
	assert.ElementsMatch(t, xy.X, expectX)
	assert.ElementsMatch(t, xy.Y, expectY)
}

func TestDesc_Populate(t *testing.T) {
	x := make([]float64, 101)
	for ind := 0; ind < len(x); ind++ {
		x[ind] = float64(100 - ind)
	}
	expectQ := []float64{0, 10, 25, 50, 75, 90, 100}
	expectMean := float64(50)
	expectN := 101
	expectS := 29.3
	d, e := NewDesc(nil, "test")
	assert.Nil(t, e)
	// first populate without sorting underlying slice
	d.Populate(x, true, nil)
	assert.ElementsMatch(t, expectQ, d.Q)
	assert.Equal(t, expectN, d.N)
	assert.Equal(t, expectMean, d.Mean)
	assert.InEpsilon(t, expectS, d.Std, .0001)
	assert.Equal(t, false, sort.Float64sAreSorted(x))

	// now populate with sorting underlying slice
	d.Populate(x, false, nil)
	assert.ElementsMatch(t, expectQ, d.Q)
	assert.Equal(t, expectN, d.N)
	assert.Equal(t, expectMean, d.Mean)
	assert.InEpsilon(t, expectS, d.Std, .0001)
	assert.Equal(t, true, sort.Float64sAreSorted(x))
}

func TestAllocRaw(t *testing.T) {
	n := 100
	x := AllocRaw(n, reflect.Float64)
	assert.Equal(t, len(x.Data), n)
	xx := make([]any, n)
	for ind := range xx {
		xx[ind] = rand.Float64()
	}
	x = NewRaw(xx, nil)
	assert.Equal(t, sort.IsSorted(x), false)
	sort.Sort(x)
	assert.Equal(t, sort.IsSorted(x), true)
}

func TestByPtr(t *testing.T) {
	x := []any{"z", "b", "a", "b", "c"}
	r := NewRaw(x, nil)
	m := ByPtr(r)
	exp := make(Levels)
	exp["a"], exp["b"], exp["c"], exp["z"] = 0, 1, 2, 3
	for k, v := range m {
		assert.Equal(t, v, exp[k])
	}
}

func TestByCounts(t *testing.T) {
	x := []any{"z", "b", "a", "b", "c", "c", "c"}
	r := NewRaw(x, nil)
	m := ByCounts(r, nil)
	exp := make(Levels)
	exp["a"], exp["b"], exp["c"], exp["z"] = 1, 2, 3, 1
	for k, v := range m {
		assert.Equal(t, v, exp[k])
	}
}

func TestLevels_TopK(t *testing.T) {
	x := []any{"z", "b", "a", "b", "c", "c", "c"}
	r := NewRaw(x, nil)
	m := ByCounts(r, nil)
	s := m.TopK(2, true, true)
	exp := `Field Value   Count
a             1
b             2
`
	assert.Equal(t, s, exp)
	s = m.TopK(2, false, false)
	exp = `Field Value   Count
c             3
b             2
`
	assert.Equal(t, s, exp)
}

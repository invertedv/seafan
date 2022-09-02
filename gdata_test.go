package seafan

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGData_AppendC(t *testing.T) {
	var e error

	gd := make(GData, 0)
	x0 := make([]any, 0)

	for ind := 0; ind < 10; ind++ {
		x0 = append(x0, float64(ind))
	}

	gd, e = gd.AppendC(NewRaw(x0, nil), "Field0", false, nil)
	assert.Nil(t, e)

	xt := []any{"a", "b", "c"}
	_, e = gd.AppendD(NewRaw(xt, nil), "Field1", nil)

	assert.NotNil(t, e)

	gd = make(GData, 0)
	gd, e = gd.AppendC(NewRaw(x0, nil), "Field0", false, nil)

	assert.Nil(t, e)

	x1 := []any{"a", "b", "c", "a", "b", "c", "a", "c", "c", "c"}
	gd, e = gd.AppendD(NewRaw(x1, nil), "Field1", nil)

	assert.Nil(t, e)

	d0 := gd.Get("Field0")

	assert.ElementsMatch(t, d0.Data, x0)
	assert.Equal(t, d0.Summary.DistrC.Mean, 4.5)

	mapx := []int32{0, 1, 2, 0, 1, 2, 0, 2, 2, 2}
	d1 := gd.Get("Field1")

	assert.ElementsMatch(t, d1.Data, mapx)

	// check normalization with supplied mean/scale works
	fp := &FParam{
		Location: 1,
		Scale:    2,
		Default:  nil,
		Lvl:      nil,
	}
	gd, e = gd.AppendC(NewRaw(x0, nil), "Field2", true, fp)

	assert.Nil(t, e)

	d2 := gd.Get("Field2")

	for ind := 0; ind < len(x0); ind++ {
		assert.Equal(t, d2.Data.([]float64)[ind], (float64(ind)-1.0)/2.0)
	}
	// Check using a supplied map works
	lvl := make(Levels)
	lvl["a"] = 2
	lvl["b"] = 3
	lvl["c"] = 4
	fp = &FParam{
		Location: 0,
		Scale:    0,
		Default:  "b",
		Lvl:      lvl,
	}
	x3 := []any{"a", "b", "c", "a", "b", "c", "a", "c", "c", "r"}
	gd, e = gd.AppendD(NewRaw(x3, nil), "Field3", fp)

	assert.Nil(t, e)

	mapx = []int32{2, 3, 4, 2, 3, 4, 2, 4, 4, 3}
	d3 := gd.Get("Field3")
	assert.ElementsMatch(t, d3.Data, mapx)
}

func TestGData_Slice(t *testing.T) {
	vecData := NewVecData("test", getData(t))
	slice, e := NewSlice("x2", 0, vecData, nil)
	assert.Nil(t, e)
	x1Exp := make([][]float64, 3)
	x1Exp[0], x1Exp[1], x1Exp[2] = []float64{1, 4, 8, 9, 10}, []float64{2}, []float64{3}
	x2Exp := []int32{0, 1, 2}

	x2OhExp := make([][]float64, 3)
	x2OhExp[0], x2OhExp[1], x2OhExp[2] = []float64{1, 0, 0}, []float64{0, 1, 0}, []float64{0, 0, 1}

	ind := 0
	for slice.Iter() {
		sl := slice.MakeSlicer()
		sliced, e := vecData.GData().Slice(sl)
		assert.Nil(t, e)
		dx1 := sliced.Get("x1").Data.([]float64)
		assert.ElementsMatch(t, x1Exp[ind], dx1)
		dx2 := sliced.Get("x2").Data.([]int32)[0]
		assert.Equal(t, dx2, x2Exp[ind])
		dx2Oh := sliced.Get("x2Oh").Data.([]float64)
		assert.ElementsMatch(t, x2OhExp[ind], dx2Oh[0:3])
		ind++
	}
}

func TestGData_Shuffle(t *testing.T) {
	rand.Seed(494949)
	gd := make(GData, 0)
	x0 := make([]any, 0)

	for ind := 0; ind < 10; ind++ {
		x0 = append(x0, float64(ind))
	}

	gd, e := gd.AppendC(NewRaw(x0, nil), "Field0", false, nil)
	assert.Nil(t, e)

	x1 := []any{"a", "b", "c", "a", "b", "c", "a", "c", "c", "c"}
	exp := []int32{0, 1, 2, 0, 1, 2, 0, 2, 2, 2}
	gd, e = gd.AppendD(NewRaw(x1, nil), "Field1", nil)
	assert.Nil(t, e)
	gd, e = gd.MakeOneHot("Field1", "Field2")
	assert.Nil(t, e)
	gd.Shuffle()

	d0 := gd.Get("Field0")
	assert.NotNil(t, d0)

	d1 := gd.Get("Field1")
	assert.NotNil(t, d1)

	d2 := gd.Get("Field2")
	assert.NotNil(t, d2)

	d0a := d0.Data.([]float64)
	d1a := d1.Data.([]int32)
	d2a := d2.Data.([]float64)
	// check that all fields moved together
	for ind := 0; ind < len(d0a); ind++ {
		indx := int32(d0a[ind])
		assert.Equal(t, exp[indx], d1a[ind])
		ohind := ind*3 + int(d1a[ind])
		assert.Condition(t, func() bool { return math.Abs(d2a[ohind]-1) < 0.0001 })
	}
}

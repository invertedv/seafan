package seafan

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGData_UpdateFts(t *testing.T) {
	var e error

	x0 := make([]any, 0)

	for ind := 0; ind < 10; ind++ {
		x0 = append(x0, float64(ind+1))
	}

	gd := NewGData()
	e = gd.AppendC(NewRaw(x0, nil), "Field0", false, nil)
	assert.Nil(t, e)

	newFt := &FType{
		Name:       "Field0",
		Role:       FRCts,
		Cats:       0,
		EmbCols:    0,
		Normalized: false,
		From:       "",
		FP:         nil,
	}
	newGd, e := gd.UpdateFts(FTypes{newFt})
	assert.Nil(t, e)
	assert.ElementsMatch(t, gd.data[0].Data.([]float64), newGd.data[0].Data.([]float64))
	gd = NewGData()
	e = gd.AppendC(NewRaw(x0, nil), "Field0", true, nil)
	assert.Nil(t, e)

	fp := &FParam{
		Location: 0,
		Scale:    1,
		Default:  nil,
		Lvl:      nil,
	}
	newFt = &FType{
		Name:       "Field0",
		Role:       FRCts,
		Cats:       0,
		EmbCols:    0,
		Normalized: true,
		From:       "",
		FP:         fp,
	}
	newGd, e = gd.UpdateFts(FTypes{newFt})
	assert.Nil(t, e)
	for ind, x := range x0 {
		assert.InEpsilon(t, x, newGd.data[0].Data.([]float64)[ind], .001)
	}
}

func TestGData_UpdateFts2(t *testing.T) {
	var e error

	gd := NewGData()

	xt := []any{"d", "e", "a", "b", "c"}
	e = gd.AppendD(NewRaw(xt, nil), "Field1", nil)
	assert.Nil(t, e)

	xt1 := []any{"e", "b", "c", "d"}
	lvls := ByPtr(NewRaw(xt1, nil))
	fp := &FParam{
		Location: 0,
		Scale:    0,
		Default:  "d",
		Lvl:      lvls,
	}
	ft := &FType{
		Name:       "Field1",
		Role:       FRCat,
		Cats:       4,
		EmbCols:    0,
		Normalized: false,
		From:       "",
		FP:         fp,
	}
	newGd, e := gd.UpdateFts(FTypes{ft})
	assert.Nil(t, e)
	result := []int32{2, 3, 2, 0, 1}
	assert.ElementsMatch(t, result, newGd.data[0].Data.([]int32))
}

func TestGData_AppendC(t *testing.T) {
	var e error

	gd := NewGData()
	x0 := make([]any, 0)

	for ind := 0; ind < 10; ind++ {
		x0 = append(x0, float64(ind))
	}

	e = gd.AppendC(NewRaw(x0, nil), "Field0", false, nil)
	assert.Nil(t, e)

	xt := []any{"a", "b", "c"}
	e = gd.AppendD(NewRaw(xt, nil), "Field1", nil)

	assert.NotNil(t, e)

	gd = NewGData()
	e = gd.AppendC(NewRaw(x0, nil), "Field0", false, nil)

	assert.Nil(t, e)

	x1 := []any{"a", "b", "c", "a", "b", "c", "a", "c", "c", "c"}
	e = gd.AppendD(NewRaw(x1, nil), "Field1", nil)

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
	e = gd.AppendC(NewRaw(x0, nil), "Field2", true, fp)

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
	e = gd.AppendD(NewRaw(x3, nil), "Field3", fp)

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
	gd := NewGData()
	x0 := make([]any, 0)

	for ind := 0; ind < 10; ind++ {
		x0 = append(x0, float64(ind))
	}

	e := gd.AppendC(NewRaw(x0, nil), "Field0", false, nil)
	assert.Nil(t, e)

	x1 := []any{"a", "b", "c", "a", "b", "c", "a", "c", "c", "c"}
	exp := []int32{0, 1, 2, 0, 1, 2, 0, 2, 2, 2}
	e = gd.AppendD(NewRaw(x1, nil), "Field1", nil)
	assert.Nil(t, e)
	e = gd.MakeOneHot("Field1", "Field2")
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

func TestGData_Sort(t *testing.T) {
	gd := NewGData()
	x0 := make([]any, 0)
	expX0 := make([]float64, 0)

	for ind := 0; ind < 10; ind++ {
		x0 = append(x0, float64(9-ind))
		expX0 = append(expX0, float64(ind))
	}

	e := gd.AppendC(NewRaw(x0, nil), "Field0", false, nil)
	assert.Nil(t, e)

	x1 := []any{"a", "b", "c", "a", "b", "c", "a", "c", "c", "c"}
	e = gd.AppendD(NewRaw(x1, nil), "Field1", nil)
	expX1 := []int32{0, 0, 0, 1, 1, 2, 2, 2, 2, 2}

	assert.Nil(t, e)

	e = gd.MakeOneHot("Field1", "Field2")

	assert.Nil(t, e)

	e = gd.Sort("Field1", true)
	assert.Nil(t, e)
	assert.ElementsMatch(t, expX1, gd.Get("Field1").Data.([]int32))

	e = gd.Sort("Field0", true)
	assert.Nil(t, e)
	assert.ElementsMatch(t, expX0, gd.Get("Field0").Data.([]float64))

	e = gd.Sort("Field2", true)
	assert.Nil(t, e)
	assert.ElementsMatch(t, expX1, gd.Get("Field1").Data.([]int32))

	e = gd.Sort("Field0", false)
	assert.Nil(t, e)
	assert.ElementsMatch(t, x0, gd.Get("Field0").Data.([]float64))
}

func TestGData_GetRaw(t *testing.T) {
	gd := NewGData()
	x0 := make([]any, 0)

	for ind := 0; ind < 10; ind++ {
		x0 = append(x0, float64(9-ind))
	}

	e := gd.AppendC(NewRaw(x0, nil), "Field0", true, nil)
	assert.Nil(t, e)
	e = gd.AppendC(NewRaw(x0, nil), "Field3", false, nil)
	assert.Nil(t, e)

	x1 := []any{"a", "b", "c", "a", "b", "c", "a", "c", "c", "c"}
	e = gd.AppendD(NewRaw(x1, nil), "Field1", nil)
	assert.Nil(t, e)

	e = gd.MakeOneHot("Field1", "Field2")

	assert.Nil(t, e)

	x1Test, e := gd.GetRaw("Field1")

	assert.Nil(t, e)
	assert.ElementsMatch(t, x1, x1Test.Data)

	x0Test, e := gd.GetRaw("Field0")
	assert.Nil(t, e)

	for ind := 0; ind < len(x0); ind++ {
		assert.Condition(t, func() bool { return math.Abs(x0[ind].(float64)-x0Test.Data[ind].(float64)) < 0.00001 })
	}

	x0Test, e = gd.GetRaw("Field3")
	assert.Nil(t, e)

	for ind := 0; ind < len(x0); ind++ {
		assert.Condition(t, func() bool { return math.Abs(x0[ind].(float64)-x0Test.Data[ind].(float64)) < 0.00001 })
	}

	x1Test, e = gd.GetRaw("Field2")

	assert.Nil(t, e)
	assert.ElementsMatch(t, x1, x1Test.Data)

}

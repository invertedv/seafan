package seafan

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"testing"

	"github.com/invertedv/chutils"

	"github.com/stretchr/testify/assert"
)

func TestGData_UpdateFts(t *testing.T) {
	var e error

	x0 := make([]any, 0)

	for ind := 0; ind < 10; ind++ {
		x0 = append(x0, float64(ind+1))
	}

	gd := NewGData()
	e = gd.AppendC(NewRaw(x0, nil), "Field0", false, nil, false)
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
	e = gd.AppendC(NewRaw(x0, nil), "Field0", true, nil, false)
	assert.Nil(t, e)

	// wrong length
	e = gd.AppendC(NewRaw(x0[0:3], nil), "fail", false, nil, false)
	assert.NotNil(t, e)

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
	e = gd.AppendD(NewRaw(xt, nil), "Field1", nil, false)
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

	e = gd.AppendC(NewRaw(x0, nil), "Field0", false, nil, false)
	assert.Nil(t, e)

	xt := []any{"a", "b", "c"}
	e = gd.AppendD(NewRaw(xt, nil), "Field1", nil, false)

	assert.NotNil(t, e)

	gd = NewGData()
	e = gd.AppendC(NewRaw(x0, nil), "Field0", false, nil, false)

	assert.Nil(t, e)

	x1 := []any{"a", "b", "c", "a", "b", "c", "a", "c", "c", "c"}
	e = gd.AppendD(NewRaw(x1, nil), "Field1", nil, false)

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
	e = gd.AppendC(NewRaw(x0, nil), "Field2", true, fp, false)

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
	e = gd.AppendD(NewRaw(x3, nil), "Field3", fp, false)

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

	e := gd.AppendC(NewRaw(x0, nil), "Field0", false, nil, false)
	assert.Nil(t, e)

	x1 := []any{"a", "b", "c", "a", "b", "c", "a", "c", "c", "c"}
	exp := []int32{0, 1, 2, 0, 1, 2, 0, 2, 2, 2}
	e = gd.AppendD(NewRaw(x1, nil), "Field1", nil, false)
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

	e := gd.AppendC(NewRaw(x0, nil), "Field0", false, nil, false)
	assert.Nil(t, e)

	x1 := []any{"a", "b", "c", "a", "b", "c", "a", "c", "c", "c"}
	e = gd.AppendD(NewRaw(x1, nil), "Field1", nil, false)
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

	e := gd.AppendC(NewRaw(x0, nil), "Field0", true, nil, false)
	assert.Nil(t, e)
	e = gd.AppendC(NewRaw(x0, nil), "Field3", false, nil, false)
	assert.Nil(t, e)

	x1 := []any{"a", "b", "c", "a", "b", "c", "a", "c", "c", "c"}
	e = gd.AppendD(NewRaw(x1, nil), "Field1", nil, false)
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

func TestGData_Read(t *testing.T) {
	var e error

	x0 := make([]any, 0)

	for ind := 0; ind < 10; ind++ {
		x0 = append(x0, float64(ind+1))
	}

	gd := NewGData()
	e = gd.AppendC(NewRaw(x0, nil), "Field0", false, nil, false)
	assert.Nil(t, e)

	x1 := []any{"a", "b", "c", "a", "b", "c", "a", "c", "c", "c"}
	e = gd.AppendD(NewRaw(x1, nil), "Field1", nil, false)
	assert.Nil(t, e)

	c, e := gd.CountLines()
	assert.Nil(t, e)
	assert.Equal(t, c, len(x0))

	row, _, e := gd.Read(1, false)
	assert.Nil(t, e)
	assert.Equal(t, row[0][0].(float64), x0[0])

	assert.Nil(t, e)
	assert.Equal(t, row[0][1].(string), x1[0])

	e = gd.Seek(4)
	assert.Nil(t, e)

	row, _, e = gd.Read(1, false)
	assert.Nil(t, e)
	assert.Equal(t, row[0][0].(float64), x0[4])

	assert.Nil(t, e)
	assert.Equal(t, row[0][1].(string), x1[4])

	ind := 0
	e = gd.Reset()
	assert.Nil(t, e)

	for {
		_, _, e = gd.Read(1, false)
		if e == io.EOF {
			break
		}
		ind++
	}
	assert.Equal(t, ind, len(x0))

	e = gd.Drop("Field1")
	assert.Nil(t, e)
	e = gd.Reset()
	assert.Nil(t, e)

	row, _, e = gd.Read(1, false)
	assert.Nil(t, e)
	assert.Equal(t, row[0][0].(float64), x0[0])
	assert.Equal(t, len(row[0]), 1)
}

func TestGData_TableSpec(t *testing.T) {
	var e error

	x0 := make([]any, 0)

	for ind := 0; ind < 10; ind++ {
		x0 = append(x0, float64(ind+1))
	}

	gd := NewGData()
	e = gd.AppendC(NewRaw(x0, nil), "Field0", false, nil, false)
	assert.Nil(t, e)

	x1 := []any{"a", "b", "c", "a", "b", "c", "a", "c", "c", "c"}
	e = gd.AppendD(NewRaw(x1, nil), "Field1", nil, false)
	assert.Nil(t, e)
	e = gd.MakeOneHot("Field1", "Field3")
	assert.Nil(t, e)

	x2 := []any{int32(0), int32(1), int32(2), int32(0), int32(1), int32(2), int32(0), int32(1), int32(2), int32(3)}
	e = gd.AppendD(NewRaw(x2, nil), "Field2", nil, false)
	assert.Nil(t, e)

	td := gd.TableSpec()
	assert.Equal(t, len(td.FieldDefs), 3)

	col, fd, e := td.Get("Field0")
	assert.Nil(t, e)
	assert.Equal(t, col, 0)
	assert.Equal(t, fd.ChSpec.Base, chutils.ChFloat)

	col, fd, e = td.Get("Field1")
	assert.Nil(t, e)
	assert.Equal(t, col, 1)
	assert.Equal(t, fd.ChSpec.Base, chutils.ChString)

	fNames := []string{"Field0", "Field1", "Field2"}
	for ind, fd := range td.FieldDefs {
		assert.Equal(t, fd.Name, fNames[ind])
	}
}

func TestGData_Join(t *testing.T) {
	gdLeft := NewGData()
	x0 := make([]any, 0)
	var e error

	for ind := 0; ind < 10; ind++ {
		x0 = append(x0, float64(ind))
	}

	e = gdLeft.AppendC(NewRaw(x0, nil), "Field0", false, nil, false)
	assert.Nil(t, e)

	xt := []any{"a", "b", "c", "a", "b", "c", "e", "f", "g", "h"}
	e = gdLeft.AppendD(NewRaw(xt, nil), "Field1", nil, false)
	assert.Nil(t, e)

	gdRight := NewGData()
	var x1 []any
	for ind := 0; ind < 5; ind++ {
		x1 = append(x1, float64(ind))
	}

	e = gdRight.AppendC(NewRaw(x1, nil), "Field2", false, nil, false)
	assert.Nil(t, e)

	xt = []any{"a", "b", "c", "k", "a"}
	e = gdRight.AppendD(NewRaw(xt, nil), "Field1", nil, false)
	assert.Nil(t, e)

	// this will not be in the joined data
	e = gdRight.MakeOneHot("Field1", "x")
	assert.Nil(t, e)

	var gdJoin *GData
	gdJoin, e = gdLeft.Join(gdRight, "Field1", Inner)
	assert.Nil(t, e)

	raw, e := gdJoin.GetRaw("Field1")
	assert.Nil(t, e)
	exp := []any{"a", "a", "a", "a", "b", "b", "c", "c"}
	assert.ElementsMatch(t, raw.Data, exp)

	exp = []any{0.0, 0.0, 3.0, 3.0, 1.0, 4.0, 2.0, 5.0}
	raw, e = gdJoin.GetRaw("Field0")
	assert.Nil(t, e)
	assert.ElementsMatch(t, raw.Data, exp)

	exp = []any{0.0, 4.0, 0.0, 4.0, 1.0, 1.0, 2.0, 2.0}
	raw, e = gdJoin.GetRaw("Field2")
	assert.Nil(t, e)
	assert.ElementsMatch(t, raw.Data, exp)

	ft := gdRight.GetFTypes().Get("Field2")
	ft.FP.Default = 110.0

	gdJoin, e = gdLeft.Join(gdRight, "Field1", Left)
	assert.Nil(t, e)

	exp = []any{"a", "a", "a", "a", "b", "b", "c", "c", "e", "f", "g", "h"}
	raw, e = gdJoin.GetRaw("Field1")
	assert.Nil(t, e)
	assert.ElementsMatch(t, raw.Data, exp)

	exp = []any{0.0, 4.0, 0.0, 4.0, 1.0, 1.0, 2.0, 2.0, 110.0, 110.0, 110.0, 110.0}
	raw, e = gdJoin.GetRaw("Field2")
	assert.Nil(t, e)
	assert.ElementsMatch(t, raw.Data, exp)

	gdJoin, e = gdLeft.Join(gdRight, "Field1", Right)
	assert.Nil(t, e)

	exp = []any{"a", "a", "a", "a", "b", "b", "c", "c", "k"}
	raw, e = gdJoin.GetRaw("Field1")
	assert.Nil(t, e)
	assert.ElementsMatch(t, raw.Data, exp)

	exp = []any{0.0, 4.0, 0.0, 4.0, 1.0, 1.0, 2.0, 2.0, 3.0}
	raw, e = gdJoin.GetRaw("Field2")
	assert.Nil(t, e)
	assert.ElementsMatch(t, raw.Data, exp)
}

// This example shows how to join two *Gdata structs.
func ExampleGData_Join() {
	// Build the first GData
	gdLeft := NewGData()

	field0 := []any{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	if e := gdLeft.AppendC(NewRaw(field0, nil), "field0", false, nil, true); e != nil {
		panic(e)
	}

	field1 := []any{"r", "s", "b", "l", "c", "s", "a"}
	if e := gdLeft.AppendD(NewRaw(field1, nil), "field1", nil, true); e != nil {
		panic(e)
	}

	field2 := []any{"A", "B", "C", "D", "E", "F", "G"}

	// value to use for field2 if gdLeft doesn't contribute to an output row
	fp := &FParam{Default: "XX"}
	if e := gdLeft.AppendD(NewRaw(field2, nil), "field2", fp, true); e != nil {
		panic(e)
	}

	// Build the second GData
	gdRight := NewGData()
	field3 := []any{100.0, 200.0, 300.0, 400.0, 500.0}
	if e := gdRight.AppendC(NewRaw(field3, nil), "field3", false, nil, true); e != nil {
		panic(e)
	}

	field1 = []any{"a", "b", "c", "k", "a"}
	if e := gdRight.AppendD(NewRaw(field1, nil), "field1", nil, true); e != nil {
		panic(e)
	}

	// do an outer join on field1
	gdJoin, err := gdLeft.Join(gdRight, "field1", Outer)
	if err != nil {
		panic(err)
	}

	for _, fld := range gdJoin.FieldList() {
		x, err := gdJoin.GetRaw(fld)
		if err != nil {
			panic(err)
		}

		fmt.Println(fld)
		fmt.Println(x.Data)
	}
	// output:
	// field0
	// [6 6 2 4 3 0 1 5 0]
	// field2
	// [G G C E D A B F XX]
	// field3
	// [100 500 200 300 0 0 0 0 400]
	// field1
	// [a a b c l r s s k]
}

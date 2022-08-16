package seafan

import (
	"github.com/stretchr/testify/assert"
	"log"
	"testing"
)

func TestGData_AppendC(t *testing.T) {
	var e error
	gd := make(GData, 0)
	x0 := make([]any, 0)
	for ind := 0; ind < 10; ind++ {
		x0 = append(x0, float64(ind))
	}
	gd, e = gd.AppendC(NewRaw(x0), "Field0", false, nil)
	if e != nil {
		log.Fatalln(e)
	}
	xt := []any{"a", "b", "c"}
	gd, e = gd.AppendD(NewRaw(xt), "Field1", nil)
	assert.NotNil(t, e)

	gd = make(GData, 0)
	gd, e = gd.AppendC(NewRaw(x0), "Field0", false, nil)
	assert.Nil(t, e)

	x1 := []any{"a", "b", "c", "a", "b", "c", "a", "c", "c", "c"}
	gd, e = gd.AppendD(NewRaw(x1), "Field1", nil)
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
	gd, e = gd.AppendC(NewRaw(x0), "Field2", true, fp)
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
	gd, e = gd.AppendD(NewRaw(x3), "Field3", fp)
	assert.Nil(t, e)
	mapx = []int32{2, 3, 4, 2, 3, 4, 2, 4, 4, 3}
	d3 := gd.Get("Field3")
	assert.ElementsMatch(t, d3.Data, mapx)

}

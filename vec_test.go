package seafan

import (
	"testing"

	"github.com/stretchr/testify/assert"
	G "gorgonia.org/gorgonia"
)

func getData(t *testing.T) *GData {
	x1 := []float64{1, 2, 3, 4, 8, 9, 10}
	x2 := []string{"a", "b", "c", "a", "a", "a", "a"}
	x3 := []int32{4, 5, 6, 1, 2, 2, 2}

	gData := NewGData()
	e := gData.AppendC(NewRawCast(x1, nil), "x1", false, nil)
	assert.Nil(t, e)

	e = gData.AppendD(NewRawCast(x2, nil), "x2", nil)
	assert.Nil(t, e)

	e = gData.AppendD(NewRawCast(x3, nil), "x3", nil)
	assert.Nil(t, e)

	e = gData.MakeOneHot("x2", "x2Oh")

	return gData
}

func TestNewVecData(t *testing.T) {
	gData := getData(t)
	vecData := NewVecData("test", gData)
	assert.Equal(t, vecData.Rows(), gData.Get("x1").Summary.NRows)
}

func TestVecData_Batch(t *testing.T) {
	vecData := NewVecData("test", getData(t))
	g := G.NewGraph()
	nd := G.NewTensor(g, G.Float64, 2, G.WithName("x1"), G.WithShape(vecData.BatchSize(), 1))
	nds := G.Nodes{nd}
	e := vecData.Init()

	assert.Nil(t, e)

	for try := 0; try < 2; try++ {
		act := make([]float64, 0)

		for vecData.Batch(nds) {
			act = append(act, nds[0].Value().Data().([]float64)...)
		}

		assert.ElementsMatch(t, act, vecData.Get("x1").Data)
	}
}

func TestSliceVecData(t *testing.T) {
	vecData := NewVecData("test", getData(t))
	slice, e := NewSlice("x2", 0, vecData, nil)
	assert.Nil(t, e)

	g := G.NewGraph()
	nd := G.NewTensor(g, G.Float64, 2, G.WithName("x1"), G.WithShape(vecData.BatchSize(), 1))
	nds := G.Nodes{nd}
	e = vecData.Init()
	assert.Nil(t, e)

	x1Exp := make([][]float64, 3)
	x1Exp[0], x1Exp[1], x1Exp[2] = []float64{1, 4, 8, 9, 10}, []float64{2}, []float64{3}

	// run through the slices
	ind := 0
	for slice.Iter() {
		sl := slice.MakeSlicer()
		newVec, e := vecData.Slice(sl)
		assert.Nil(t, e)
		x1act := make([]float64, 0)
		// run through the batches, accumulate x1
		for newVec.Batch(nds) {
			x1act = append(x1act, nds[0].Value().Data().([]float64)...)
		}

		assert.ElementsMatch(t, x1Exp[ind], x1act)

		ind++
	}
}

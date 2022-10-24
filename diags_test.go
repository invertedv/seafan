package seafan

import (
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCoalesce(t *testing.T) {
	obs := []float64{0, 0, 1,
		0, 0, 1,
		0, 1, 0,
		1, 0, 0}
	fit := []float64{.2, .3, .5,
		.2, .5, .3,
		.2, .4, .4,
		.5, .3, .2}
	nCat := 3

	expObs := []float64{1, 1, 0, 0}
	expFit := []float64{.5, .3, .4, .2}

	targ := []int{2}
	fitTest, e := Coalesce(fit, nCat, targ, false, false, nil)
	assert.Nil(t, e)

	obsTest, e := Coalesce(obs, nCat, targ, true, false, nil)
	assert.Nil(t, e)

	assert.ElementsMatch(t, obsTest, expObs)
	assert.ElementsMatch(t, fitTest, expFit)

	targ = []int{1, 2}
	expObs = []float64{1, 1, 1, 0}
	expFit = []float64{.8, .8, .8, .5}

	fitTest, e = Coalesce(fit, nCat, targ, false, false, nil)
	assert.Nil(t, e)

	obsTest, e = Coalesce(obs, nCat, targ, true, false, nil)
	assert.Nil(t, e)

	assert.Nil(t, e)
	assert.ElementsMatch(t, obsTest, expObs)
	assert.ElementsMatch(t, fitTest, expFit)
}

func TestKS(t *testing.T) {
	y := make([]float64, 0)
	p := make([]float64, 0)
	n := 1000
	cnt := 0

	for k := 0; k < n; k++ {
		y = append(y, 0, 1)
		px := float64(k) / float64(n)
		p = append(p, 1.0-px, px)
		y = append(y, 1, 0)
		p = append(p, 1-px*px, px*px)
		if px*px <= 0.25 {
			cnt++
		}
	}

	fit, e := Coalesce(p, 2, []int{1}, false, false, nil)
	assert.Nil(t, e)

	obs, e := Coalesce(y, 2, []int{1}, true, false, nil)
	assert.Nil(t, e)

	xy, e := NewXY(fit, obs)

	assert.Nil(t, e)
	ks, _, _, e := KS(xy, nil)

	assert.Nil(t, e)
	assert.InEpsilon(t, ks, 25.0, .01)
}

func ExampleSlice_Iter() {
	// An example of slicing through the data to generate diagnostics on subsets.
	// The code here will generate a decile plot for each of the 20 levels of x4.
	Verbose = false
	bSize := 100
	// generate a Pipeline of type *ChData that reads test.csv in the data directory
	pipe := chPipe(bSize, "test1.csv")
	// The feature x4 takes on values 0,1,2,...19.  chPipe treats this a continuous feature.
	// Let's override that and re-initialize the pipeline.

	WithCats("x4")(pipe)
	WithOneHot("x4oh", "x4")(pipe)

	if e := pipe.Init(); e != nil {
		panic(e)
	}

	mod := ModSpec{
		"Input(x1+x2+x3+x4oh)",
		"FC(size:2, activation:softmax)",
		"Target(yoh)",
	}
	nn, e := NewNNModel(mod, pipe, true)

	if e != nil {
		panic(e)
	}
	WithCostFn(CrossEntropy)(nn)

	ft := NewFit(nn, 100, pipe)

	if e = ft.Do(); e != nil {
		panic(e)
	}

	sf := os.TempDir() + "/nnTest"
	e = nn.Save(sf)

	if e != nil {
		panic(e)
	}

	WithBatchSize(8500)(pipe)

	pred, e := PredictNN(sf, pipe, false)
	if e != nil {
		panic(e)
	}

	if e = AddFitted(pipe, sf, []int{1}, "fit", nil, false, nil); e != nil {
		panic(e)
	}

	_ = os.Remove(sf + "P.nn")
	_ = os.Remove(sf + "S.nn")

	s, e := NewSlice("x4", 0, pipe, nil)
	if e != nil {
		panic(e)
	}

	fit, e := Coalesce(pred.FitSlice(), 2, []int{1}, false, false, nil)
	if e != nil {
		panic(e)
	}
	desc, e := NewDesc(nil, "Descriptive Statistics")

	for s.Iter() {
		slicer := s.MakeSlicer()
		if e != nil {
			panic(e)
		}
		desc.Populate(fit, true, slicer)
		fmt.Printf("Slice x4=%v has %d observations\n", s.Value(), desc.N)
	}
	// Output:
	// Slice x4=0 has 391 observations
	// Slice x4=1 has 408 observations
	// Slice x4=2 has 436 observations
	// Slice x4=3 has 428 observations
	// Slice x4=4 has 417 observations
	// Slice x4=5 has 472 observations
	// Slice x4=6 has 424 observations
	// Slice x4=7 has 455 observations
	// Slice x4=8 has 431 observations
	// Slice x4=9 has 442 observations
	// Slice x4=10 has 411 observations
	// Slice x4=11 has 413 observations
	// Slice x4=12 has 433 observations
	// Slice x4=13 has 416 observations
	// Slice x4=14 has 434 observations
	// Slice x4=15 has 367 observations
	// Slice x4=16 has 437 observations
	// Slice x4=17 has 433 observations
	// Slice x4=18 has 429 observations
	// Slice x4=19 has 423 observations
}

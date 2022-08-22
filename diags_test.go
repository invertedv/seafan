package seafan

import (
	"github.com/stretchr/testify/assert"
	"testing"
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
	xy, e := Coalesce(obs, fit, nCat, targ, false, nil)
	assert.Nil(t, e)
	assert.ElementsMatch(t, xy.Y, expObs)
	assert.ElementsMatch(t, xy.X, expFit)

	targ = []int{1, 2}
	expObs = []float64{1, 1, 1, 0}
	expFit = []float64{.8, .8, .8, .5}
	xy, e = Coalesce(obs, fit, nCat, targ, false, nil)
	assert.Nil(t, e)
	assert.ElementsMatch(t, xy.Y, expObs)
	assert.ElementsMatch(t, xy.X, expFit)
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
	ks, _, _, e := KS(y, p, 2, []int{1}, false, nil, nil)
	assert.Nil(t, e)
	assert.InEpsilon(t, ks, 25.0, .01)
}

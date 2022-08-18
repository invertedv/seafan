package seafan

import (
	"fmt"
	"github.com/invertedv/chutils"
	"github.com/invertedv/chutils/file"
	"github.com/stretchr/testify/assert"
	"log"
	"os"
	"testing"
)

func chPipe(bSize int) *ChData {
	dataPath := os.Getenv("data") // path to data directory
	fileName := dataPath + "/test1.csv"
	f, e := os.Open(fileName)
	if e != nil {
		log.Fatalln(e)
	}
	// set up chutils file reader
	rdr := file.NewReader(fileName, ',', '\n', 0, 0, 1, 0, f, 0)
	e = rdr.Init("", chutils.MergeTree)
	if e != nil {
		log.Fatalln(e)
	}
	// determine data types
	e = rdr.TableSpec().Impute(rdr, 0, .99)
	if e != nil {
		log.Fatalln(e)
	}
	ch := NewChData("Test ch Pipeline", WithBatchSize(bSize),
		WithReader(rdr), WithCycle(true),
		WithCats("y", "y1", "y2"),
		WithOneHot("yoh", "y"),
		WithOneHot("y1oh", "y1"))
	// initialize pipeline
	e = ch.Init()
	if e != nil {
		log.Fatalln(e)
	}
	return ch
}
func TestFit_Do(t *testing.T) {

	bSize := 100
	pipe := chPipe(bSize)
	mod, e := ByFormula("yoh~x1+x2+x3+x4", pipe)
	assert.Nil(t, e)
	nn := NewNNModel(bSize, mod, nil, WithCostFn(CrossEntropy))
	epochs := 150
	ft := NewFit(nn, epochs, pipe)
	e = ft.Do()
	assert.Nil(t, e)
	for _, n := range nn.Params() {
		fmt.Println(n.Name(), n.Value().Data().([]float64))
	}
	wts := []float64{-2.06, -3.5, 1, -0.08} //glm logistic estimates
	n := nn.G().ByName("lWeightsOut").Nodes()[0].Value().Data().([]float64)
	for ind, w := range wts {
		assert.InEpsilon(t, n[ind], w, .15)
	}
}

func ExampleFit_Do() {
	bSize := 100
	// generate a Pipeline of type *ChData that reads test.csv in the data directory
	pipe := chPipe(bSize)
	// model: target and features.  Target yoh is one-hot with 2 levels
	mod, e := ByFormula("yoh~x1+x2+x3+x4", pipe)
	if e != nil {
		log.Fatalln(e)
	}
	// model is straight-forward with no hidden layers or dropouts.
	nn := NewNNModel(bSize, mod, nil, WithCostFn(CrossEntropy))
	epochs := 150
	ft := NewFit(nn, epochs, pipe)
	e = ft.Do()
	if e != nil {
		log.Fatalln(e)
	}
	// Plot the in-sample cost in a browser (default: firefox)
	e = ft.InCosts().Plot(&PlotDef{Title: "In Sample Cost Curve", Height: 1200, Width: 1200, Show: true, XTitle: "epoch", YTitle: "Cost"}, true)
	if e != nil {
		log.Fatalln(e)
	}
	// Output:
	// rows read:  8500
	// best epoch:  150
	// elapsed time 0.1 minutes
}

package seafan

import (
	"fmt"
	"github.com/invertedv/chutils"
	"github.com/invertedv/chutils/file"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/stat"
	"log"
	"math"
	"os"
	"testing"
)

func chPipe(bSize int, fileName string) *ChData {
	dataPath := os.Getenv("data") // path to data directory
	f, e := os.Open(dataPath + "/" + fileName)
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

func TestNNModel_Save(t *testing.T) {
	Verbose = false
	pipe := chPipe(100, "test1.csv")
	mod := ModSpec{
		"Input(x1+x2+x3+x4)",
		"FC(size:2, activation:softmax)",
		"Output(yoh)",
	}
	//
	nn, e := NewNNModel(mod, pipe, true)
	assert.Nil(t, e)
	WithCostFn(CrossEntropy)(nn)
	e = nn.Save("/home/will/tmp/testnn")
	assert.Nil(t, e)
	exp := make([]float64, 0)
	for _, n := range nn.paramsW {
		x := n.Nodes()[0].Value().Data().([]float64)
		for ind := 0; ind < len(x); ind++ {
			exp = append(exp, math.Round(x[ind]*100.0)/100.0)
		}
	}
	nn1, e := LoadNN("/home/will/tmp/testnn", pipe, false)
	assert.Nil(t, e)
	act := make([]float64, 0)
	for _, n := range nn1.paramsW {
		x := n.Nodes()[0].Value().Data().([]float64)
		for ind := 0; ind < len(x); ind++ {
			act = append(act, math.Round(x[ind]*100.0)/100.0)
		}
	}
	assert.ElementsMatch(t, exp, act)
	assert.ElementsMatch(t, mod, nn.construct)
}

func TestFit_Do(t *testing.T) {
	Verbose = false
	pipe := chPipe(100, "test1.csv")
	mod := ModSpec{
		"Input(x1+x2+x3+x4)",
		"FC(size:2, activation:softmax)",
		"Output(yoh)",
	}
	nn, e := NewNNModel(mod, pipe, true)
	assert.Nil(t, e)
	WithCostFn(CrossEntropy)(nn)
	epochs := 150
	ft := NewFit(nn, epochs, pipe)
	e = ft.Do()
	assert.Nil(t, e)
	wts := []float64{-2.06, -3.5, 1, -0.08} //glm logistic estimates
	n := nn.G().ByName("lWeights1").Nodes()[0].Value().Data().([]float64)
	for ind, w := range wts {
		assert.InEpsilon(t, n[ind], w, .15)
	}
}

func ExampleWithOneHot() {
	// This example shows a model that incorporates a feature (x4) as one-hot and an embedding
	Verbose = false
	bSize := 100
	// generate a Pipeline of type *ChData that reads test.csv in the data directory
	pipe := chPipe(bSize, "test1.csv")
	// The feature x4 takes on values 0,1,2,...19.  chPipe treats this a continuous feature.
	// Let's override that and re-initialize the pipeline.
	WithCats("x4")(pipe)
	WithOneHot("x4oh", "x4")(pipe)

	if e := pipe.Init(); e != nil {
		log.Fatalln(e)
	}
	mod := ModSpec{
		"Input(x1+x2+x3+x4oh)",
		"FC(size:2, activation:softmax)",
		"Output(yoh)",
	}
	//
	fmt.Println("x4 as one-hot")
	nn, e := NewNNModel(mod, pipe, true)
	if e != nil {
		log.Fatalln(e)
	}
	fmt.Println(nn)
	fmt.Println("x4 as embedding")
	mod = ModSpec{
		"Input(x1+x2+x3+E(x4oh,3))",
		"FC(size:2, activation:softmax)",
		"Output(yoh)",
	}
	nn, e = NewNNModel(mod, pipe, true)
	if e != nil {
		log.Fatalln(e)
	}
	fmt.Println(nn)
	// Output:
	//x4 as one-hot
	//
	//Inputs
	//Field x1
	//	continuous
	//
	//Field x2
	//	continuous
	//
	//Field x3
	//	continuous
	//
	//Field x4oh
	//	one-hot
	//	derived from feature x4
	//	length 20
	//
	//Target
	//Field yoh
	//	one-hot
	//	derived from feature y
	//	length 2
	//
	//Model Structure
	//Input(x1+x2+x3+x4oh)
	//FC(size:2, activation:softmax)
	//Output(yoh)
	//
	//Batch size: 100
	//
	//x4 as embedding
	//
	//Inputs
	//Field x1
	//	continuous
	//
	//Field x2
	//	continuous
	//
	//Field x3
	//	continuous
	//
	//Field x4oh
	//	embedding
	//	derived from feature x4
	//	length 20
	//	embedding dimension of 3
	//
	//Target
	//Field yoh
	//	one-hot
	//	derived from feature y
	//	length 2
	//
	//Model Structure
	//Input(x1+x2+x3+E(x4oh,3))
	//FC(size:2, activation:softmax)
	//Output(yoh)
	//
	//Batch size: 100

}

func ExampleWithOneHot_example2() {
	// This example incorporates a drop out layer
	Verbose = false
	bSize := 100
	// generate a Pipeline of type *ChData that reads test.csv in the data directory
	pipe := chPipe(bSize, "test1.csv")
	// generate model: target and features.  Target yoh is one-hot with 2 levels
	mod := ModSpec{
		"Input(x1+x2+x3+x4)",
		"FC(size:3, activation:relu)",
		"DropOut(.1)",
		"FC(size:2, activation:softmax)",
		"Output(yoh)",
	}

	nn, e := NewNNModel(mod, pipe, true,
		WithCostFn(CrossEntropy),
		WithName("Example With Dropouts"))
	if e != nil {
		log.Fatalln(e)
	}
	fmt.Println(nn)
	// Output:
	//Example With Dropouts
	//Inputs
	//Field x1
	//	continuous
	//
	//Field x2
	//	continuous
	//
	//Field x3
	//	continuous
	//
	//Field x4
	//	continuous
	//
	//Target
	//Field yoh
	//	one-hot
	//	derived from feature y
	//	length 2
	//
	//Model Structure
	//Input(x1+x2+x3+x4)
	//FC(size:3, activation:relu)
	//DropOut(.1)
	//FC(size:2, activation:softmax)
	//Output(yoh)
	//
	//Cost function: CrossEntropy
	//
	//Batch size: 100
}

func ExampleFit_Do() {
	Verbose = false
	bSize := 100
	// generate a Pipeline of type *ChData that reads test.csv in the data directory
	pipe := chPipe(bSize, "test1.csv")
	// generate model: target and features.  Target yoh is one-hot with 2 levels
	mod := ModSpec{
		"Input(x1+x2+x3+x4)",
		"FC(size:3, activation:relu)",
		"DropOut(.1)",
		"FC(size:2, activation:softmax)",
		"Output(yoh)",
	}
	// model is straight-forward with no hidden layers or dropouts.
	nn, e := NewNNModel(mod, pipe, true, WithCostFn(CrossEntropy))
	if e != nil {
		log.Fatalln(e)
	}
	epochs := 150
	ft := NewFit(nn, epochs, pipe)
	e = ft.Do()
	if e != nil {
		log.Fatalln(e)
	}
	// Plot the in-sample cost in a browser (default: firefox)
	e = ft.InCosts().Plot(&PlotDef{Title: "In-Sample Cost Curve", Height: 1200, Width: 1200, Show: true, XTitle: "epoch", YTitle: "Cost"}, true)
	if e != nil {
		log.Fatalln(e)
	}
	// Output:
}

func ExampleFit_Do_example2() {
	// This example demonstrates how to use a validation sample for early stopping
	Verbose = false
	bSize := 100
	// generate a Pipeline of type *ChData that reads test.csv in the data directory
	mPipe := chPipe(bSize, "test1.csv")
	vPipe := chPipe(1000, "testVal.csv")

	// generate model: target and features.  Target yoh is one-hot with 2 levels
	mod := ModSpec{
		"Input(x1+x2+x3+x4)",
		"FC(size:3, activation:relu)",
		"DropOut(.1)",
		"FC(size:2, activation:softmax)",
		"Output(yoh)",
	}
	nn, e := NewNNModel(mod, mPipe, true, WithCostFn(CrossEntropy))
	if e != nil {
		log.Fatalln(e)
	}
	epochs := 150
	ft := NewFit(nn, epochs, mPipe)
	WithValidation(vPipe, 10)(ft)
	e = ft.Do()
	if e != nil {
		log.Fatalln(e)
	}
	// Plot the in-sample cost in a browser (default: firefox)
	e = ft.InCosts().Plot(&PlotDef{Title: "In-Sample Cost Curve", Height: 1200, Width: 1200, Show: true, XTitle: "epoch", YTitle: "Cost"}, true)
	if e != nil {
		log.Fatalln(e)
	}
	e = ft.OutCosts().Plot(&PlotDef{Title: "Validation Sample Cost Curve", Height: 1200, Width: 1200, Show: true, XTitle: "epoch", YTitle: "Cost"}, true)
	if e != nil {
		log.Fatalln(e)
	}
	// Output:
}

func ExamplePredictNN() {
	// This example demonstrates fitting a regression model and predicting on new data
	Verbose = false
	bSize := 100
	// generate a Pipeline of type *ChData that reads test.csv in the data directory
	mPipe := chPipe(bSize, "test1.csv")
	vPipe := chPipe(1000, "testVal.csv")

	// This model is OLS
	mod := ModSpec{
		"Input(x1+x2+x3+x4)",
		"FC(size:1)",
		"Output(ycts)",
	}
	// model is straight-forward with no hidden layers or dropouts.
	nn, e := NewNNModel(mod, mPipe, true, WithCostFn(RMS))
	if e != nil {
		log.Fatalln(e)
	}
	epochs := 150
	ft := NewFit(nn, epochs, mPipe)
	e = ft.Do()
	if e != nil {
		log.Fatalln(e)
	}
	sf := os.TempDir() + "/nnTest"
	e = nn.Save(sf)
	if e != nil {
		log.Fatalln(e)
	}
	pred, e := PredictNN(sf, vPipe, false)
	if e != nil {
		log.Fatalln(e)
	}
	fmt.Printf("out-of-sample correlation: %0.2f\n", stat.Correlation(pred.FitSlice(), pred.ObsSlice(), nil))
	_ = os.Remove(sf + "P.nn")
	if e != nil {
		log.Fatalln(e)
	}
	_ = os.Remove(sf + "S.nn")
	// Output:
	// out-of-sample correlation: 0.84

}

func ExampleWithCallBack() {
	// This example shows how to create a callback during the fitting phase (fit.Do).
	// The callback is called at the end of each epoch.  The callback below loads a new dataset after
	// epoch 100.

	Verbose = false
	bSize := 100
	// generate a Pipeline of type *ChData that reads test.csv in the data directory
	mPipe := chPipe(bSize, "test1.csv")
	// This callback function replaces the initial dataset with newData.csv after epoch 2500
	cb := func(c Pipeline) {
		switch d := c.(type) {
		case *ChData:
			if d.Epoch(-1) == 100 {
				dataPath := os.Getenv("data") // path to data directory
				fileName := dataPath + "/testVal.csv"
				f, e := os.Open(fileName)
				if e != nil {
					log.Fatalln(e)
				}
				rdrx := file.NewReader(fileName, ',', '\n', 0, 0, 1, 0, f, 0)
				if e := rdrx.Init("", chutils.MergeTree); e != nil {
					log.Fatalln(e)
				}
				if e := rdrx.TableSpec().Impute(rdrx, 0, .99); e != nil {
					log.Fatalln(e)
				}
				rows, _ := rdrx.CountLines()
				fmt.Println("New data at end of epoch ", d.Epoch(-1))
				fmt.Println("Number of rows ", rows)
				WithReader(rdrx)(d)

			}
		}
	}
	WithCallBack(cb)(mPipe)

	// This model is OLS
	mod := ModSpec{
		"Input(x1+x2+x3+x4)",
		"FC(size:1)",
		"Output(ycts)",
	}
	// model is straight-forward with no hidden layers or dropouts.
	nn, e := NewNNModel(mod, mPipe, true, WithCostFn(RMS))
	if e != nil {
		log.Fatalln(e)
	}
	epochs := 150
	ft := NewFit(nn, epochs, mPipe)
	e = ft.Do()
	if e != nil {
		log.Fatalln(e)
	}
	// Output:
	//New data at end of epoch  100
	//Number of rows  1000
}

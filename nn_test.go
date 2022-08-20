package seafan

import (
	"fmt"
	"github.com/invertedv/chutils"
	"github.com/invertedv/chutils/file"
	"github.com/stretchr/testify/assert"
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
	pipe := chPipe(100, "test1.csv")
	mod := ModSpec{
		"Input(x1,x2,x3,x4)",
		"Dropout(.11)",
		"FC(size:3, activation:leakyrelu(0.1))",
		"Dropout(.1)",
		"FC(size:2)",
		"Dropout(.1)",
		"FC(size:2,bias:false, activation:softmax)",
		"Output(yoh)",
	}
	//
	//ft, e := mod.Output(pipe)
	nn, e := NewNNModel(mod, pipe)
	assert.Nil(t, e)
	e = nn.Save("/home/will/tmp/testnn")
	assert.Nil(t, e)
	exp := make([]float64, 0)
	for _, n := range nn.paramsW {
		fmt.Println(n.Nodes()[0].Name(), n.Nodes()[0].Shape())
		x := n.Nodes()[0].Value().Data().([]float64)
		for ind := 0; ind < len(x); ind++ {
			exp = append(exp, math.Round(x[ind]*100.0)/100.0)
		}
	}
	nn1, e := LoadNN("/home/will/tmp/testnn", pipe, false)
	assert.Nil(t, e)
	fmt.Println("reading")
	act := make([]float64, 0)
	for _, n := range nn1.paramsW {
		fmt.Println(n.Nodes()[0].Name(), n.Nodes()[0].Shape())
		x := n.Nodes()[0].Value().Data().([]float64)
		for ind := 0; ind < len(x); ind++ {
			act = append(act, math.Round(x[ind]*100.0)/100.0)
		}
	}
	assert.ElementsMatch(t, exp, act)
	assert.ElementsMatch(t, mod, nn.construct)
	fmt.Println(nn)
}

/*
func TestFit_Do(t *testing.T) {
	Verbose = false
	bSize := 100
	pipe := chPipe(bSize, "test1.csv")
	mod, e := ByFormula("yoh~x1+x2+x3+x4", pipe)
	assert.Nil(t, e)
	nn := NewNNModel(bSize, mod, nil, WithCostFn(CrossEntropy))
	epochs := 150
	ft := NewFit(nn, epochs, pipe)
	e = ft.Do()
	assert.Nil(t, e)
	wts := []float64{-2.06, -3.5, 1, -0.08} //glm logistic estimates
	n := nn.G().ByName("lWeightsOut").Nodes()[0].Value().Data().([]float64)
	for ind, w := range wts {
		assert.InEpsilon(t, n[ind], w, .15)
	}
}

*/
/*
func ExampleWithOneHot() {
	// This example shows a model that incorporates a feature (x4) as one-hot and an embedding
	Verbose = false
	bSize := 100
	// generate a Pipeline of type *ChData that reads test.csv in the data directory
	pipe := chPipe(bSize, "test1.csv")
	// The feature x4 takes on values 0,1,2,...19.  chPipe treats this a a continuous feature.
	// Let's override that and re-initialize the pipeline.
	WithCats("x4")(pipe)
	WithOneHot("x4oh", "x4")(pipe)

	if e := pipe.Init(); e != nil {
		log.Fatalln(e)
	}

	mod, e := ByFormula("yoh~x1+x2+x3+x4oh", pipe)
	if e != nil {
		log.Fatalln(e)
	}
	fmt.Println("x4 as one-hot")
	nn := NewNNModel(bSize, mod, nil, WithCostFn(CrossEntropy))
	fmt.Println(nn)
	fmt.Println("x4 as embedding")
	mod, e = ByFormula("yoh~x1+x2+x3+E(x4oh,3)", pipe)
	if e != nil {
		log.Fatalln(e)
	}
	nn = NewNNModel(bSize, mod, nil, WithCostFn(CrossEntropy))
	fmt.Println(nn)
	// Output:
	//x4 as one-hot
	//
	//Inputs:
	//	Field x1
	//		continuous
	//	Field x2
	//		continuous
	//	Field x3
	//		continuous
	//	Field x4oh
	//		one-hot
	//		derived from feature x4
	//		length 20
	//
	// Target:
	//	Field yoh
	//		one-hot
	//		derived from feature y
	//		length 2
	//
	// Cost function: CrossEntropy
	// Batch size: 100
	// NN structure:
	//	FCLayer Layer 0: (23, 1) (output)
	//
	//
	// x4 as embedding
	//
	// Inputs:
	//	Field x1
	//		continuous
	//	Field x2
	//		continuous
	//	Field x3
	//		continuous
	//	Field x4oh
	//		embedding
	//		derived from feature x4
	//		length 20
	//		embedding dimension of 3
	//
	// Target:
	//	Field yoh
	//		one-hot
	//		derived from feature y
	//		length 2
	//
	// Cost function: CrossEntropy
	// Batch size: 100
	// NN structure:
	//	FCLayer Layer 0: (6, 1) (output)

}

*/
/*
func ExampleWithDropOuts() {
	Verbose = false
	bSize := 100
	// generate a Pipeline of type *ChData that reads test.csv in the data directory
	pipe := chPipe(bSize, "test1.csv")
	// generate model: target and features.  Target yoh is one-hot with 2 levels
	mod, e := ByFormula("yoh~x1+x2+x3+x4", pipe)
	if e != nil {
		log.Fatalln(e)
	}
	// model is straight-forward with no hidden layers or dropouts.
	drops := Drops{
		&Drop{AfterLayer: 1, DropProb: .1},
		&Drop{AfterLayer: 2, DropProb: .05},
	}
	nn := NewNNModel(bSize, mod, []int{3, 4, 2},
		WithCostFn(CrossEntropy),
		WithDropOuts(drops),
		WithName("Example With Dropouts"))
	fmt.Println(nn)
	// Output:
	// Example With Dropouts
	// Inputs:
	//	Field x1
	//		continuous
	//	Field x2
	//		continuous
	//	Field x3
	//		continuous
	//	Field x4
	//		continuous
	//
	// Target:
	//	Field yoh
	//		one-hot
	//		derived from feature y
	//		length 2
	//
	// Cost function: CrossEntropy
	// Batch size: 100
	// NN structure:
	//	FCLayer Layer 0: (4, 3)
	//	Drop Layer (probability = 0.10)
	//	FCLayer Layer 1: (3, 4)
	//	Drop Layer (probability = 0.05)
	//	FCLayer Layer 2: (4, 2)
	//	FCLayer Layer 3: (2, 1) (output)
}

*/
/*
func ExampleFit_Do() {
	Verbose = false
	bSize := 100
	// generate a Pipeline of type *ChData that reads test.csv in the data directory
	pipe := chPipe(bSize, "test1.csv")
	// generate model: target and features.  Target yoh is one-hot with 2 levels
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
	e = ft.InCosts().Plot(&PlotDef{Title: "In-Sample Cost Curve", Height: 1200, Width: 1200, Show: true, XTitle: "epoch", YTitle: "Cost"}, true)
	if e != nil {
		log.Fatalln(e)
	}
	// Output:
}

*/
/*
func ExampleFit_Do_example2() {
	// This example demonstrates how to use a validation sample for early stopping
	Verbose = false
	bSize := 100
	// generate a Pipeline of type *ChData that reads test.csv in the data directory
	mPipe := chPipe(bSize, "test1.csv")
	vPipe := chPipe(1000, "testVal.csv")

	// generate model: target and features.  Target yoh is one-hot with 2 levels
	mod, e := ByFormula("yoh~x1+x2+x3+x4", mPipe)
	if e != nil {
		log.Fatalln(e)
	}
	// model is straight-forward with no hidden layers or dropouts.
	nn := NewNNModel(bSize, mod, nil, WithCostFn(CrossEntropy))
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

*/

/*
func ExamplePredictNN() {
	// This example demonstrates fitting a regression model and predicting on new data
	Verbose = false
	bSize := 100
	// generate a Pipeline of type *ChData that reads test.csv in the data directory
	mPipe := chPipe(bSize, "test1.csv")
	vPipe := chPipe(1000, "testVal.csv")

	// generate model: target and features.  Target yoh is one-hot with 2 levels
	mod, e := ByFormula("ycts~x1+x2+x3+x4", mPipe)
	if e != nil {
		log.Fatalln(e)
	}
	// model is straight-forward with no hidden layers or dropouts.
	nn := NewNNModel(bSize, mod, nil, WithCostFn(RMS))
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
	pred, e := PredictNN(sf, vPipe.BatchSize(), vPipe)
	if e != nil {
		log.Fatalln(e)
	}
	fmt.Printf("out-of-sample correlation: %0.2f\n", stat.Correlation(pred.FitSlice(), pred.ObsSlice(), nil))
	e = os.Remove(sf + "P.nn")
	if e != nil {
		log.Fatalln(e)
	}
	e = os.Remove(sf + "S.nn")
	// Output:
	// out-of-sample correlation: 0.84

}


*/

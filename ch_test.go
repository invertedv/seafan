package seafan

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/invertedv/chutils"
	"github.com/invertedv/chutils/file"
	"github.com/stretchr/testify/assert"
	G "gorgonia.org/gorgonia"
)

func TestChData_Init(t *testing.T) {
	dataPath := os.Getenv("data")
	fileName := dataPath + "/test1.csv"
	f, e := os.Open(fileName)

	assert.Nil(t, e)

	rdr := file.NewReader(fileName, ',', '\n', 0, 0, 1, 0, f, 0)
	e = rdr.Init("", chutils.MergeTree)

	assert.Nil(t, e)

	e = rdr.TableSpec().Impute(rdr, 0, .99)

	assert.Nil(t, e)

	bsize := 100
	ch := NewChData("Test ch Pipeline", WithBatchSize(bsize),
		WithReader(rdr), WithCycle(true),
		WithCats("y", "y1", "y2", "x4"),
		WithOneHot("yoh", "y"),
		WithOneHot("y1oh", "y1"),
		WithOneHot("x4oh", "x4"),
		WithNormalized("x1", "x2", "x3"),
		WithOneHot("y2oh", "y2"))
	e = ch.Init()

	assert.Nil(t, e)

	expRows := 8500

	assert.Equal(t, expRows, ch.Rows())
	assert.Equal(t, bsize, ch.BatchSize())

	// check roles and # of categories
	roles := make(map[string]FRole)
	roles["x1"], roles["yoh"], roles["x4"] = FRCts, FROneHot, FRCat
	cats := make(map[string]int)
	cats["x1"], cats["yoh"], cats["x4"] = 0, 2, 20
	ind := 0

	for k, v := range roles {
		ft := ch.GetFType(k)
		assert.NotNil(t, ft)
		assert.Equal(t, v, ft.Role)
		assert.Equal(t, ft.Cats, cats[k])
		ind++
	}

	// check correctly normalized
	ftX1 := ch.Get("x1")

	assert.InEpsilon(t, ftX1.Summary.DistrC.Std, 1.0, 0.0001)
	assert.Condition(t, func() bool { return math.Abs(ftX1.Summary.DistrC.Mean) < 0.0001 })

	m := ftX1.FT.FP.Location // mean of data (not normalized)
	// set the normalization of x1 to location=42 and scale=1
	fts := ch.GetFTypes()
	ft := fts.Get("x1")
	ft.FP.Scale = 1
	ft.FP.Location = 42

	// set the FTypes so these values will be used
	WithFtypes(fts)(ch)
	e = ch.Init()

	assert.Nil(t, e)
	assert.InEpsilon(t, m-42.0, ch.Get("x1").Summary.DistrC.Mean, 0.0001)
}

func TestChData_Batch(t *testing.T) {
	dataPath := os.Getenv("data")
	fileName := dataPath + "/test1.csv"
	f, e := os.Open(fileName)

	assert.Nil(t, e)

	rdr := file.NewReader(fileName, ',', '\n', 0, 0, 1, 0, f, 0)
	e = rdr.Init("", chutils.MergeTree)

	assert.Nil(t, e)

	e = rdr.TableSpec().Impute(rdr, 0, .99)

	assert.Nil(t, e)

	bsize := 100
	ch := NewChData("Test ch Pipeline", WithBatchSize(bsize),
		WithReader(rdr), WithCycle(true),
		WithCats("y", "y1", "y2", "x4"),
		WithOneHot("yoh", "y"),
		WithOneHot("y1oh", "y1"),
		WithOneHot("x4oh", "x4"),
		WithNormalized("x1", "x2", "x3"),
		WithOneHot("y2oh", "y2"))
	e = ch.Init()

	assert.Nil(t, e)

	g := G.NewGraph()
	node := G.NewTensor(g, G.Float64, 2, G.WithName("x1"), G.WithShape(bsize, 1), G.WithInit(G.Zeroes()))

	// run through batchs and verify counts and mean of x1 is zero
	sumX := 0.0
	n := 0
	for ch.Batch(G.Nodes{node}) {
		n += bsize
		x := node.Value().Data().([]float64)
		for _, xv := range x {
			sumX += xv
		}
	}
	mean := sumX / float64(n)

	assert.Equal(t, n, 8500)
	assert.Condition(t, func() bool { return math.Abs(mean) < 0.0001 })
}

func ExampleChData_Init() {
	dataPath := os.Getenv("data") // path to data directory
	fileName := dataPath + "/test1.csv"
	f, e := os.Open(fileName)

	if e != nil {
		panic(e)
	}

	// set up chutils file reader
	rdr := file.NewReader(fileName, ',', '\n', 0, 0, 1, 0, f, 0)
	e = rdr.Init("", chutils.MergeTree)
	if e != nil {
		panic(e)
	}

	// determine data types
	e = rdr.TableSpec().Impute(rdr, 0, .99)

	if e != nil {
		panic(e)
	}

	bSize := 100
	ch := NewChData("Test ch Pipeline", WithBatchSize(bSize),
		WithReader(rdr), WithCycle(true),
		WithCats("y", "y1", "y2", "x4"),
		WithOneHot("yoh", "y"),
		WithOneHot("y1oh", "y1"),
		WithOneHot("x4oh", "x4"),
		WithNormalized("x1", "x2", "x3"),
		WithOneHot("y2oh", "y2"))
	// initialize pipeline
	e = ch.Init()

	if e != nil {
		panic(e)
	}
	// Output:
	// rows read:  8500
}

func ExampleChData_Batch() {
	dataPath := os.Getenv("data") // path to data directory
	fileName := dataPath + "/test1.csv"
	f, e := os.Open(fileName)

	if e != nil {
		panic(e)
	}
	// set up chutils file reader
	rdr := file.NewReader(fileName, ',', '\n', 0, 0, 1, 0, f, 0)
	e = rdr.Init("", chutils.MergeTree)

	if e != nil {
		panic(e)
	}

	// determine data types
	e = rdr.TableSpec().Impute(rdr, 0, .99)

	if e != nil {
		panic(e)
	}

	bSize := 100
	ch := NewChData("Test ch Pipeline",
		WithBatchSize(bSize),
		WithReader(rdr),
		WithNormalized("x1"))
	// create a graph & node to illustrate Batch()
	g := G.NewGraph()
	node := G.NewTensor(g, G.Float64, 2, G.WithName("x1"), G.WithShape(bSize, 1), G.WithInit(G.Zeroes()))

	var sumX = 0.0
	n := 0
	// run through batchs and verify counts and mean of x1 is zero
	for ch.Batch(G.Nodes{node}) {
		n += bSize
		x := node.Value().Data().([]float64)
		for _, xv := range x {
			sumX += xv
		}
	}

	mean := sumX / float64(n)

	fmt.Printf("mean of x1: %0.2f", math.Abs(mean))
	// Output:
	// rows read:  8500
	// mean of x1: 0.00
}

func ExampleChData_Batch_example2() {
	// We can normalize fields by values we supply rather than the values in the epoch.
	dataPath := os.Getenv("data") // path to data directory
	fileName := dataPath + "/test1.csv"
	f, e := os.Open(fileName)

	if e != nil {
		panic(e)
	}

	// set up chutils file reader
	rdr := file.NewReader(fileName, ',', '\n', 0, 0, 1, 0, f, 0)
	e = rdr.Init("", chutils.MergeTree)

	if e != nil {
		panic(e)
	}

	// determine data types
	e = rdr.TableSpec().Impute(rdr, 0, .99)

	if e != nil {
		panic(e)
	}

	bSize := 100
	// Let's normalize x1 with location=41 and scale=1
	ft := &FType{
		Name:       "x1",
		Role:       0,
		Cats:       0,
		EmbCols:    0,
		Normalized: true,
		From:       "",
		FP:         &FParam{Location: 40, Scale: 1},
	}
	ch := NewChData("Test ch Pipeline",
		WithBatchSize(bSize),
		WithReader(rdr))

	WithFtypes(FTypes{ft})(ch)

	// create a graph & node to illustrate Batch()
	g := G.NewGraph()
	node := G.NewTensor(g, G.Float64, 2, G.WithName("x1"), G.WithShape(bSize, 1), G.WithInit(G.Zeroes()))

	sumX := 0.0
	n := 0
	// run through batchs and verify counts and mean of x1 is zero
	for ch.Batch(G.Nodes{node}) {
		n += bSize
		x := node.Value().Data().([]float64)
		for _, xv := range x {
			sumX += xv
		}
	}

	mean := sumX / float64(n)

	fmt.Printf("mean of x1: %0.2f", math.Abs(mean))
	// Output:
	// rows read:  8500
	// mean of x1: 39.50
}

func ExampleChData_SaveFTypes() {
	// Field Types (FTypes) can be saved once they're created.  This preserves key information like
	//  - The field role
	//  - Location and Scale used in normalization
	//  - Mapping of discrete fields
	//  - Construction of one-hot fields
	dataPath := os.Getenv("data") // path to data directory
	fileName := dataPath + "/test1.csv"
	f, e := os.Open(fileName)

	if e != nil {
		panic(e)
	}

	// set up chutils file reader
	rdr := file.NewReader(fileName, ',', '\n', 0, 0, 1, 0, f, 0)
	e = rdr.Init("", chutils.MergeTree)

	if e != nil {
		panic(e)
	}

	// determine data types
	e = rdr.TableSpec().Impute(rdr, 0, .99)

	if e != nil {
		panic(e)
	}

	bSize := 100
	ch := NewChData("Test ch Pipeline", WithBatchSize(bSize),
		WithReader(rdr), WithCycle(true),
		WithCats("y", "y1", "y2", "x4"),
		WithOneHot("yoh", "y"),
		WithOneHot("y1oh", "y1"),
		WithOneHot("x4oh", "x4"),
		WithNormalized("x1", "x2", "x3"),
		WithOneHot("y2oh", "y2"))
	// initialize pipeline
	e = ch.Init()

	if e != nil {
		panic(e)
	}

	outFile := os.TempDir() + "/seafan.json"

	if e = ch.SaveFTypes(outFile); e != nil {
		panic(e)
	}

	saveFTypes, e := LoadFTypes(outFile)

	if e != nil {
		panic(e)
	}

	ch1 := NewChData("Saved FTypes", WithReader(rdr), WithBatchSize(bSize),
		WithFtypes(saveFTypes))

	if e := ch1.Init(); e != nil {
		panic(e)
	}

	fmt.Printf("Role of field y1oh: %s", ch.GetFType("y1oh").Role)
	// Output:
	// rows read:  8500
	// rows read:  8500
	// Role of field y1oh: FROneHot
}

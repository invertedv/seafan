package seafan

import (
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/invertedv/utilities"

	"github.com/stretchr/testify/assert"

	"github.com/invertedv/chutils"
	s "github.com/invertedv/chutils/sql"
)

// Simple date arithmetic is possible.  The function dateAdd(d,m) adds m months to d.
// The data is:
// row, newField1, newField2, newField3, date
// 0,row0,.1,.2, 3/1/2023
// 2,row2,2.1,3.2, 4/1/2023
// 3,row3,3.1,4.2, 5/1/2023
// 4,row4,4.1,5.2, 6/1/2023
// 1,row1,1.1,2.2, 7/1/2023
// 100,row100,4100.0,5200.0, 8/1/2020
func ExampleEvaluate_dateAdd() {
	Verbose = false
	var (
		outPipe Pipeline
		err     error
	)

	data := os.Getenv("data")
	pipe, e := CSVToPipe(data+"/pipeTest2.csv", nil, false)
	if e != nil {
		panic(e)
	}

	root := &OpNode{Expression: "dateAdd(date,row)"}

	if err = Expr2Tree(root); err != nil {
		panic(err)
	}

	if err = Evaluate(root, pipe); err != nil {
		panic(err)
	}

	if outPipe, err = AddToPipe(root, "nextMonth", pipe); err != nil {
		panic(err)
	}

	fmt.Println("date + row months")

	raw, e := outPipe.GData().GetRaw("nextMonth")
	if e != nil {
		panic(e)
	}

	fmt.Println(raw.Data)
	// output:
	// date + row months
	// [2023-03-01 00:00:00 +0000 UTC 2023-06-01 00:00:00 +0000 UTC 2023-08-01 00:00:00 +0000 UTC 2023-10-01 00:00:00 +0000 UTC 2023-08-01 00:00:00 +0000 UTC 2028-12-01 00:00:00 +0000 UTC]
}

func ExampleEvaluate_if() {
	var (
		outPipe Pipeline
		err     error
	)

	Verbose = false

	data := os.Getenv("data")
	pipe, e := CSVToPipe(data+"/pipeTest2.csv", nil, false)
	if e != nil {
		panic(e)
	}

	root := &OpNode{Expression: "if(date=='3/1/2023',1,0)"}

	if err = Expr2Tree(root); err != nil {
		panic(err)
	}

	if err = Evaluate(root, pipe); err != nil {
		panic(err)
	}

	if outPipe, err = AddToPipe(root, "march2023", pipe); err != nil {
		panic(err)
	}

	fmt.Println(pipe.Get("march2023").Data.([]float64))
	root = &OpNode{Expression: "if(date>'3/1/2023',1,0)"}

	if err = Expr2Tree(root); err != nil {
		panic(err)
	}

	if err = Evaluate(root, pipe); err != nil {
		panic(err)
	}

	if outPipe, err = AddToPipe(root, "afterMarch2023", pipe); err != nil {
		panic(err)
	}
	fmt.Println(outPipe.Get("afterMarch2023").Data.([]float64))
	// output:
	// [1 0 0 0 0 0]
	// [0 1 1 1 1 0]
}

func TestEvaluate_date(t *testing.T) {
	Verbose = false
	dataCF := "'3/25/2022', '20230228' " // c
	dataDr := "0.1, .2 "                 // D
	dataPV := "'0','0'"                  // e
	pipe := buildPipe([]string{dataCF, dataDr, dataPV}, []string{"s", "f", "s"})
	exprs := "toDate(c)"
	results := []any{any(time.Date(2022, 3, 25, 0, 0, 0, 0, time.UTC)),
		any(time.Date(2023, 2, 28, 0, 0, 0, 0, time.UTC))}
	root := &OpNode{Expression: exprs}
	if err := Expr2Tree(root); err != nil {
		panic(err)
	}

	e := Evaluate(root, pipe)
	assert.Nil(t, e)
	assert.ElementsMatch(t, root.Raw.Data, results)
}

// tests conditional statements with strings
func TestExpr2Tree(t *testing.T) {
	Verbose = false
	dataCF := "'0', 'b', '0', 'd'" // c
	dataDr := "0.1, .2, .3, .6"    // D
	dataPV := "'0','0','0','abc'"  // e
	pipe := buildPipe([]string{dataCF, dataDr, dataPV}, []string{"s", "f", "s"})
	exprs := []string{"c=='b'", "c=='0'", "c==e", "e=='abc'", "c!=D", "c*2", "D==.1", "e+'a'", "c > 'b'", "c>='b'", "log(c)"}
	results := [][]float64{{0, 1, 0, 0}, {1, 0, 1, 0}, {1, 0, 1, 0}, {0, 0, 0, 1}, {0, 0, 0, 0},
		{0, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 0, 0}}
	errs := []bool{false, false, false, false, true, true, false, true, false, false, true}

	for ind, expr := range exprs {
		root := &OpNode{Expression: expr}

		if err := Expr2Tree(root); err != nil {
			panic(err)
		}

		e := Evaluate(root, pipe)
		if errs[ind] {
			assert.NotNil(t, e)
			continue
		}

		assert.Nil(t, e)
		assert.ElementsMatch(t, root.Raw.Data, results[ind])
		if ind == 0 {
			break
		}
	}
}

// Test CopyNode
func TestCopyNode(t *testing.T) {
	Verbose = false
	dataCF := "1, 2, 3, 4"      // c
	dataDr := "0.1, .2, .3, .4" // D
	dataPV := "6,0,0,0"         // e
	pipe := buildPipe([]string{dataCF, dataDr, dataPV}, []string{"f", "f", "f"})

	root := &OpNode{Expression: "log(e+c+1)*2+5"}
	if err := Expr2Tree(root); err != nil {
		panic(err)
	}

	newNode := CopyNode(root)

	e := Evaluate(root, pipe)
	assert.Nil(t, e)
	assert.Nil(t, newNode.Raw) // no value to newNode yet

	e = Evaluate(newNode, pipe)
	assert.Nil(t, e)
	assert.ElementsMatch(t, root.Raw.Data, newNode.Raw.Data) // now they should match
}

// test irr and npv functions
func TestEvalSFunction(t *testing.T) {
	Verbose = false
	dataCF := "1, 2, 3, 4"      // c
	dataDr := "0.1, .2, .3, .4" // D
	dataPV := "6,0,0,0"         // e
	dataS := "'x','a','z','t'"  // f
	pipe := buildPipe([]string{dataCF, dataDr, dataPV, dataS}, []string{"f", "f", "f", "s"})

	pvx := tester("max(f)", pipe)
	assert.Equal(t, pvx[0].(string), "z")

	expIr := 0.3169080407719
	ir := tester("irr(e,c)", pipe)
	assert.InDelta(t, ir[0], expIr, .0001, nil)

	// constant discount rate
	pv := tester("npv(.1,c)", pipe)
	ExpVal := 8.302778
	assert.InDelta(t, pv[0], ExpVal, .0001)

	// variable discount rate
	pv = tester("npv(D,c)", pipe)
	ExpVal = 5.8995
	assert.InDelta(t, pv[0], ExpVal, .0001)
}

// TestEvaluate2 should all generate errors
func TestEvaluate2(t *testing.T) {
	Verbose = false
	dataCF := "1, 2, 3, 4"      // c
	dataDr := "0.1, .2, .3, .4" // D
	dataPV := "6,0,0,0"         // e
	dataS := "'x','a','z','t'"  // f
	pipe := buildPipe([]string{dataCF, dataDr, dataPV, dataS}, []string{"f", "f", "f", "s"})

	express := []string{"c+f", "f+1", "f*f", "index(c,f)", "c^f", "log(e)"}

	for _, exp := range express {
		root := &OpNode{Expression: exp}
		err := Expr2Tree(root)
		assert.Nil(t, err)

		err = Evaluate(root, pipe)
		assert.NotNil(t, err)
	}
}

// TestEvaulate4 tests Lag
func TestEvaluate_lag(t *testing.T) {
	Verbose = false
	dataC := "1, 2"
	dataD := "'20230228', '20230301'"
	pipe := buildPipe([]string{dataC, dataD}, []string{"f", "s"})
	exprs := []string{"lag(c,3)", "lag(D,3)"}
	results := [][]any{{3.0, 1.0}, {"3.00", "20230228"}}

	for ind, expr := range exprs {
		act := tester(expr, pipe)
		assert.ElementsMatch(t, results[ind], act)
	}
}

// TestEvaluate_toFloat_cat tests toFloat, and cat()
func TestEvaluate_toFloat_cat(t *testing.T) {
	var (
		outPipe Pipeline
		err     error
	)

	Verbose = false
	dataC := "1, 2"
	dataD := "'34', '50'"
	pipe := buildPipe([]string{dataC, dataD}, []string{"f", "s"})

	exprs := []string{"toFloat(c)"}
	results := [][]any{{1.0, 2.0}}
	for ind, expr := range exprs {
		act := tester(expr, pipe)
		assert.ElementsMatch(t, results[ind], act)
	}

	root := &OpNode{Expression: "cat(c)"}
	err = Expr2Tree(root)
	assert.Nil(t, err)

	err = Evaluate(root, pipe)
	assert.Nil(t, err)
	assert.Equal(t, root.Role, FRCat)

	outPipe, err = AddToPipe(root, "catval", pipe)
	assert.Nil(t, err)

	assert.Equal(t, outPipe.Get("catval").FT.Role, FRCat)
}

// TestEvaluate3 tests toString and toDate functions
func TestEvaluate_toString(t *testing.T) {
	Verbose = false
	dataC := "1, 2"
	dataD := "'20230228', '20230301'"
	pipe := buildPipe([]string{dataC, dataD}, []string{"f", "s"})

	exprs := []string{"toString(cat(c))", "toString(c)", "toString(toDate(D))"}
	results := [][]string{{"1", "2"}, {"1.00", "2.00"}, {"2/28/2023", "3/1/2023"}}
	for ind, expr := range exprs {
		act := tester(expr, pipe)
		assert.ElementsMatch(t, results[ind], act)
	}
}

func TestEvaluate_range(t *testing.T) {
	var (
		outPipe Pipeline
		err     error
	)

	Verbose = false
	dataC := "1"
	dataD := "30"
	pipe := buildPipe([]string{dataC, dataD}, []string{"f", "f"})

	root := &OpNode{Expression: "range(0,10)"}
	if err = Expr2Tree(root); err != nil {
		panic(err)
	}

	err = Evaluate(root, pipe)
	assert.Nil(t, err)

	outPipe, err = AddToPipe(root, "range", pipe)
	assert.Nil(t, err)

	assert.Equal(t, outPipe.Rows(), 10)
}

func TestEvaluate(t *testing.T) {
	Verbose = false
	dataC := "1, 2"
	dataD := "3, 10"
	pipe := buildPipe([]string{dataC, dataD}, []string{"f", "f", "f"})

	frmla := []string{
		"sum(c) - npv(.1,D)",
		"if(c==1.0,D==3.0,c)",
		"count(c)",
		"min(c)",
		"prodAfter(D)",
		"prodBefore(D)",
		"lag(c,42)",
		"c+D",
		"cumeBefore(c)",
		"if(c==1,log(c),-c)",
		"max(c)",
		"c-D-D",
		"row(c)",
		"index(D,1-(c-1))",
		"countBefore(c)",
		"c-D-D",
		"-D*3 + D",
		"lag(c,42)",
		"countBefore(c)",
		"cumeBefore(c)",
		"countAfter(c)",
		"cumeAfter(c)",
		"std(c)",
		"max(c)",
		//		"median(c)",
		"mean(-c)",
		"sum(c+D)",
		"sum(c)",
		"if(c-1.5,1,0)",
		"-D*3 + D",
		"-D + 4*c",
		"-(D ^ (c-1))",
		"log(c)*(c-2) + D",
		"(((-(c))))",
		"c >=3 || D==10",
		"if(c>=2 || D==3, 1, 0)",
		"if(c==1,log(c),-c)",
		"c+3*D",
		"(c-D)*(c+D)",
		"if(c>1,c,D)",
		"if(c>1 && D>2,1,0)",
		"-(c+3)*(D-3)",
		"if(c==1,log(c),c)",
	}

	expect := [][]float64{
		{-9.09090909090909},
		{1, 2},
		{2},
		{1},
		{30, 10},
		{3, 30},
		{42, 1},
		{4, 12},
		{1, 3},
		{0, -2},
		{2},
		{-5, -18},
		{0, 1},
		{10, 3},
		{1, 2},
		{-5, -18},
		{-6, -20},
		{42, 1},
		{1, 2},
		{1, 3},
		{2, 1},
		{3, 2},
		{0.7071067811865476},
		{2},
		//		{1},
		{-1.5},
		{16},
		{3},
		{0, 1},
		{-6, -20},
		{1, -2},
		{-1, -10},
		{3, 10},
		{-1, -2},
		{0, 1},
		{1, 1},
		{0, -2},
		{10, 32},
		{-8, -96},
		{3, 2},
		{0, 1},
		{0, -35},
		{0, 2},
	}

	for ind := 0; ind < len(frmla); ind++ {
		act := tester(frmla[ind], pipe)
		acts := make([]float64, len(act))
		for indx, a := range act {
			r, e := utilities.Any2Kind(a, reflect.Float64)
			assert.Nil(t, e)
			acts[indx] = r.(float64)
		}
		assert.EqualValues(t, expect[ind], acts)
	}
}

func TestLoop(t *testing.T) {
	Verbose = false

	expect := [][]float64{
		{6, 20},
		{-3, -17},
		{4, 5},
	}
	dataC := "1, 2"
	dataD := "3, 10"
	pipe := buildPipe([]string{dataC, dataD}, []string{"f", "f", "f"})
	eqns := []string{"D*x", "1-r+x", "c+x"}
	assign := []string{"r", "y", "c"}
	ops := make([]*OpNode, 0)

	for ind := 0; ind < len(eqns); ind++ {
		op := &OpNode{Expression: eqns[ind]}
		if e := Expr2Tree(op); e != nil {
			panic(e)
		}
		ops = append(ops, op)
	}

	if e := Loop("x", 1, 3, ops, assign, pipe); e != nil {
		panic(e)
	}
	for ind := 0; ind < len(assign); ind++ {
		act := pipe.Get(assign[ind]).Data.([]float64)
		assert.EqualValues(t, expect[ind], act)
	}
}

func buildPipe(data, types []string) Pipeline {
	var sel, arrjoin []string
	outCols := "cDefg"
	inCols := "stuvw"

	for ind := 0; ind < len(data); ind++ {
		data[ind] = fmt.Sprintf("array(%s) AS %s", data[ind], inCols[ind:ind+1])
		if types[ind] == "f" {
			sel = append(sel, fmt.Sprintf("toFloat64(%s) AS %s", inCols[ind:ind+1], outCols[ind:ind+1]))
		}
		if types[ind] == "s" {
			sel = append(sel, fmt.Sprintf("%s AS %s", inCols[ind:ind+1], outCols[ind:ind+1]))
		}
		arrjoin = append(arrjoin, fmt.Sprintf(inCols[ind:ind+1]))
	}
	qry := fmt.Sprintf("WITH d AS (SELECT %s) SELECT %s FROM d ARRAY JOIN %s", strings.Join(data, ","), strings.Join(sel, ","), strings.Join(arrjoin, ","))

	user := os.Getenv("user")
	pw := os.Getenv("pw")
	conn, err := chutils.NewConnect("127.0.0.1", user, pw, nil)
	if err != nil {
		panic(err)
	}

	rdr := s.NewReader(qry, conn)

	if e := rdr.Init("", chutils.MergeTree); e != nil {
		panic(e)
	}

	pipe := NewChData("model run")
	WithReader(rdr)(pipe)
	WithBatchSize(0)(pipe)
	if e := pipe.Init(); e != nil {
		panic(e)
	}

	return pipe
}

func tester(eqn string, pipe Pipeline) []any {
	root := &OpNode{Expression: eqn}
	if err := Expr2Tree(root); err != nil {
		panic(err)
	}

	if e := Evaluate(root, pipe); e != nil {
		panic(e)
	}

	return root.Raw.Data
}

// We'll add two fields to the pipeline: the sum=c+D and max=max(c,D)
func ExampleAddToPipe() {
	var (
		outPipe Pipeline
		err     error
	)

	Verbose = false

	// builds a Pipline with two fields:
	//    c = 1,2
	//    D = 3,-4
	pipe := buildPipe([]string{"1,2", "3,-4"}, []string{"f", "f"})
	// we'll add two fields to the pipeline: the sum=c+d and max=max(c,d)

	// start by parsing the expressions.
	field1 := &OpNode{Expression: "c+D"}
	if e := Expr2Tree(field1); e != nil {
		panic(e)
	}

	field2 := &OpNode{Expression: "if(c>D,c,D)"}
	if e := Expr2Tree(field2); e != nil {
		panic(e)
	}
	// field1 and field2 nodes now have the structure of the expressions

	// evaluate these on pipe
	if err = Evaluate(field1, pipe); err != nil {
		panic(err)
	}

	if err = Evaluate(field2, pipe); err != nil {
		panic(err)
	}

	// now add them to pipe
	if outPipe, err = AddToPipe(field1, "sum", pipe); err != nil {
		panic(err)
	}

	if outPipe, err = AddToPipe(field2, "max", outPipe); err != nil {
		panic(err)
	}

	// see what we got
	field1Val := outPipe.Get("sum")
	fmt.Println(field1Val.Data.([]float64))

	field2Val := outPipe.Get("max")
	fmt.Println(field2Val.Data.([]float64))

	// Output:
	// [4 -2]
	// [3 2]
}

// In this example, we calculate four expressions and return them to the pipeline.
// The field c is added to itself each iteration.
// The field r stores the loop field.  On return, it will have the last value of the loop.
//
// The fields are evaulated in order, starting with the 0 element of inner.
func ExampleLoop() {
	Verbose = false

	// builds a Pipline with two fields:
	//    c = 1,2,3,4
	//    D = 5,-5,3,6
	pipe := buildPipe([]string{"1,2,3,4", "5,-5,3,6"}, []string{"f", "f"})
	// we'll add two fields to the pipeline: the sum=c+d and max=max(c,d)

	// start by parsing the expressions.
	field1, result1 := &OpNode{Expression: "c+c"}, "c"
	field2, result2 := &OpNode{Expression: "indx"}, "r" // indx will be the name of the looping field.
	field3, result3 := &OpNode{Expression: "D*c"}, "s"
	field4, result4 := &OpNode{Expression: "s-1"}, "t"

	inner := []*OpNode{field1, field2, field3, field4}
	assign := []string{result1, result2, result3, result4}

	for ind := 0; ind < len(assign); ind++ {
		if e := Expr2Tree(inner[ind]); e != nil {
			panic(e)
		}
	}

	if e := Loop("indx", 1, 3, inner, assign, pipe); e != nil {
		panic(e)
	}

	for ind := 0; ind < len(assign); ind++ {
		fmt.Println(assign[ind])
		fmt.Println(pipe.Get(assign[ind]).Data.([]float64))
	}
	// output:
	// c
	// [4 8 12 16]
	// r
	// [2 2 2 2]
	// s
	// [20 -40 36 96]
	// t
	// [19 -41 35 95]
}

// This example shows how to print a result
func ExampleEvaluate() {
	Verbose = false

	// builds a Pipline with two fields:
	//    c = 1,2,3,4
	//    D = 5,-5,3,6
	pipe := buildPipe([]string{"1,2,3,4", "'a', 'b', 'c', 'd'"}, []string{"f", "s"})

	field := &OpNode{Expression: "print(c, 0)"}
	if e := Expr2Tree(field); e != nil {
		panic(e)
	}

	if e := Evaluate(field, pipe); e != nil {
		panic(e)
	}
	// output:
	// c
	// 0: 1
	// 1: 2
	// 2: 3
	// 3: 4
}

// This is an examples of plots
func ExampleEvaluate_2() {
	Verbose = false

	pipe := buildPipe([]string{"1,2,3,4", "6,7,8,9", "9,8,7,6", "1,2,1,1"}, []string{"f", "f", "f", "f"})

	cmds := []string{"setPlotDim(500,300)",
		"histogram(f,'green', 'counts')",
		"render('', 'Histogram', 'Data','Counts')",
		"newPlot()",
		"plotXY(c,D,'line','black')",
		"render('','One Line','X label','Y label')",
		"plotXY(c,e,'markers','red')",
		"render('','Two Lines','X label','Y label')",
		"newPlot()",
		"plotXY(c,e,'line','blue')",
		"setPlotDim(1000,1000)",
		"render('','One Line Again','X label','Y label')",
		"newPlot()",
		"plotLine(D,'line','green')",
		"plotLine(D,'markers','yellow')",
		"render('','plotLine Test','Auto-x', 'Y')",
	}

	for ind := 0; ind < len(cmds); ind++ {
		node := &OpNode{Expression: cmds[ind]}

		if e := Expr2Tree(node); e != nil {
			panic(e)
		}

		if e := Evaluate(node, pipe); e != nil {
			panic(e)
		}
	}

	// output:

}

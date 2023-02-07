package seafan

import (
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/invertedv/chutils"
	s "github.com/invertedv/chutils/sql"
)

func TestEvaluate(t *testing.T) {
	Verbose = false
	dataC := "1, 2"
	dataD := "3, 10"
	pipe := buildPipe(dataC, dataD)

	frmla := []string{
		"index(D,1-(c-1))",
		"row(c)",
		"c-D-D",
		"-D*3 + D",
		"lag(c,42)",
		"countb(c)",
		"cumb(c,42)",
		"counta(c)",
		"cuma(c, 42)",
		"c-D-D",
		"s(c)",
		"max(c)",
		"median(c)",
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
		{10, 3},
		{0, 1},
		{-5, -18},
		{-6, -20},
		{42, 1},
		{0, 1},
		{42, 1},
		{1, 0},
		{2, 42},
		{-5, -18},
		{0.7071067811865476},
		{2},
		{1},
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
		assert.EqualValues(t, expect[ind], act)
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
	pipe := buildPipe(dataC, dataD)
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

func buildPipe(data1, data2 string) Pipeline {
	qry := "WITH d AS (SELECT array(%s) AS a, array(%s) AS b) SELECT toInt32(a) AS c, toInt32(b) AS D FROM d ARRAY JOIN a,b"
	qry = fmt.Sprintf(qry, data1, data2)

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

func tester(eqn string, pipe Pipeline) []float64 {
	root := &OpNode{Expression: eqn}
	if err := Expr2Tree(root); err != nil {
		panic(err)
	}

	if e := Evaluate(root, pipe); e != nil {
		return nil
	}

	return root.Value
}

// We'll add two fields to the pipeline: the sum=c+D and max=max(c,D)
func ExampleAddToPipe() {
	Verbose = false

	// builds a Pipline with two fields:
	//    c = 1,2
	//    D = 3,-4
	pipe := buildPipe("1,2", "3,-4")
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
	if e1 := Evaluate(field1, pipe); e1 != nil {
		panic(e1)
	}

	if e1 := Evaluate(field2, pipe); e1 != nil {
		panic(e1)
	}

	// now add them to pipe
	if e := AddToPipe(field1, "sum", pipe); e != nil {
		panic(e)
	}

	if e := AddToPipe(field2, "max", pipe); e != nil {
		panic(e)
	}

	// see what we got
	field1Val := pipe.Get("sum")
	fmt.Println(field1Val.Data.([]float64))

	field2Val := pipe.Get("max")
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
	pipe := buildPipe("1,2,3,4", "5,-5,3,6")
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

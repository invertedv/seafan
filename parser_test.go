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

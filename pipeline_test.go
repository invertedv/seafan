package seafan

import (
	"fmt"
	"os"
)

// Create a Pipeline from a CSV and force a specific FType.
// The values of the field "row" are integers: 1,2,3,4,5,6,7
// If we just load the CSV, row will be treated as float64 (continuous).
// The field ft instructs the code to treat it as categorical.
func ExampleCSVToPipe() {
	Verbose = false

	// row takes on values 1,2,3,...  If we do nothing, the pipe will convert these to float64.
	// Specifying the role as FRCat will cause "row" to be treated as categorical.
	ft := &FType{
		Name:       "row",
		Role:       FRCat,
		Cats:       0,
		EmbCols:    0,
		Normalized: false,
		From:       "",
		FP:         nil,
	}

	data := os.Getenv("data") + "/pipeTest1.csv"
	pipe, e := CSVToPipe(data, FTypes{ft})
	if e != nil {
		panic(e)
	}

	fmt.Println("# Rows: ", pipe.Rows())
	mapped := pipe.Get("row").Data.([]int32)
	fmt.Println(mapped)
	// categorical values are mapped to int32.
	fmt.Println("\nmap for field row:")
	rowMap := pipe.GetFType("row").FP.Lvl
	// the raw values in pipeTest1.csv run from 1 to 7
	for raw := int64(1); raw < int64(len(mapped))+1; raw++ {
		fmt.Printf("raw: %v, mapped: %v\n", raw, rowMap[any(raw)])
	}
	// output:
	// # Rows:  7
	// [0 1 2 3 4 5 6]
	//
	// map for field row:
	// raw: 1, mapped: 0
	// raw: 2, mapped: 1
	// raw: 3, mapped: 2
	// raw: 4, mapped: 3
	// raw: 5, mapped: 4
	// raw: 6, mapped: 5
	// raw: 7, mapped: 6
}

// This example shows how to join two pipelines on a common field.
// The Pipelines pipe1 and pipe2 share a field "row" that has values 1, 2, 3, 4.
//
// In the pipelines pipe1 has values of row: 1,..,7 and pipe2 has values of row: 0,..,4.
func ExampleJoin() {
	Verbose = false

	// row takes on values 1,2,3,...  If we do nothing, the pipe will convert these to float64.
	// Specifying the role as FRCat will cause "row" to be treated as categorical.
	ft := &FType{
		Name:       "row",
		Role:       FRCat,
		Cats:       0,
		EmbCols:    0,
		Normalized: false,
		From:       "",
		FP:         nil,
	}
	data := os.Getenv("data")
	pipe2, e := CSVToPipe(data+"/pipeTest2.csv", FTypes{ft})
	if e != nil {
		panic(e)
	}

	// sort pipe2 by join field
	if e := pipe2.GData().Sort("row", true); e != nil {
		panic(e)
	}

	pipe1, e := CSVToPipe(data+"/pipeTest1.csv", FTypes{ft})
	if e != nil {
		panic(e)
	}
	joinPipe, e := Join(pipe1, pipe2, "row")
	if e != nil {
		panic(e)
	}

	fmt.Println("# Rows: ", joinPipe.Rows())

	raw, e := joinPipe.GData().GetRaw("row")
	if e != nil {
		panic(e)
	}

	fmt.Println("common row values: ", raw.Data)
	// output:
	// # Rows:  4
	// common row values:  [1 2 3 4]
}

// This example shows how to join two pipelines on a common field, in this case a date.
func ExampleJoin_dateJoin() {
	Verbose = false

	// In this example, we don't need to specify an FType since dates will be FRCat.
	data := os.Getenv("data")
	pipe2, e := CSVToPipe(data+"/pipeTest2.csv", nil)
	if e != nil {
		panic(e)
	}

	pipe1, e := CSVToPipe(data+"/pipeTest3.csv", nil)
	if e != nil {
		panic(e)
	}
	joinPipe, e := Join(pipe1, pipe2, "date")
	if e != nil {
		fmt.Println("oops -- not sorted")
	}

	// sort pipe2 by join field
	if err := pipe2.GData().Sort("date", true); err != nil {
		panic(err)
	}

	joinPipe, e = Join(pipe1, pipe2, "date")
	if e != nil {
		panic(e)
	}

	raw, e := joinPipe.GData().GetRaw("date")
	if e != nil {
		panic(e)
	}

	fmt.Println("# Rows: ", joinPipe.Rows())
	fmt.Println("common date values: ", raw.Data)
	// output:
	// oops -- not sorted
	// # Rows:  2
	// common date values:  [2023-04-01 00:00:00 +0000 UTC 2023-05-01 00:00:00 +0000 UTC]
}

// This example shows another way to accomplish a join if the join field is already in the pipe and is
// not FRCat.  The parser function cat() converts a FRCts field to FRCat.
func ExampleJoin_cat() {
	Verbose = false

	data := os.Getenv("data")
	pipe2, e := CSVToPipe(data+"/pipeTest2.csv", nil)
	if e != nil {
		panic(e)
	}

	// set up parser to execute a function converted field "row" to categorical
	root := &OpNode{Expression: "cat(row)"}

	if err := Expr2Tree(root); err != nil {
		panic(err)
	}

	if err := Evaluate(root, pipe2); err != nil {
		panic(err)
	}

	if err := AddToPipe(root, "rowCat", pipe2); err != nil {
		panic(err)
	}

	// now sort pipe by our new field
	if e := pipe2.GData().Sort("rowCat", true); e != nil {
		panic(e)
	}

	pipe1, e := CSVToPipe(data+"/pipeTest1.csv", nil)
	if e != nil {
		panic(e)
	}

	// create the field in the next pipe, too.
	if err := Evaluate(root, pipe1); err != nil {
		panic(err)
	}

	if err := AddToPipe(root, "rowCat", pipe1); err != nil {
		panic(err)
	}

	// note, the field "row" is not being joined on but is in both pipes -- need to drop it from one of them
	pipe1.GData().Drop("row")

	joinPipe, e := Join(pipe1, pipe2, "rowCat")
	if e != nil {
		panic(e)
	}

	fmt.Println("# Rows: ", joinPipe.Rows())

	raw, e := joinPipe.GData().GetRaw("row")
	if e != nil {
		panic(e)
	}

	fmt.Println("common row values: ", raw.Data)
	// output:
	// # Rows:  4
	// common row values:  [1 2 3 4]
}

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

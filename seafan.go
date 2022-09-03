// Package seafan is a set of tools for building DNN modes. The build engine is gorgonia (https://pkg.go.dev/gorgonia.org/gorgonia).
//
// Seafan features:
//
// - A data pipeline based on chutils (https://github.com/invertedv/chutils) to access files and ClickHouse tables.
//   - Point-and-shoot specification of the data
//   - Simple specification of one-hot features
//
// - A wrapper around gorgonia that meshes to the pipeline.
//   - Simple specification of models, including embeddings
//   - A fit method with optional early stopping and callbacks
//   - Saving and loading models
//
// - Model diagnostics for categorical targets.
//   - KS plots
//   - Decile plots
//
// - Utilities.
//   - Plotting wrapper for plotly (https://github.com/MetalBlueberry/go-plotly) for xy plots.
//   - Numeric struct for (x,y) data and plotting and descriptive statistics.
package seafan

import "fmt"

// Verbose controls amount of printing.
var Verbose = true

// Browser is the browser to use for plotting.
var Browser = "firefox"

type SeaError int

const (
	ErrPipe SeaError = 0 + iota
	ErrData
	ErrFields
	ErrGData
	ErrChData
	ErrModSpec
	ErrNNModel
	ErrDiags
	ErrVecData
)

func (seaErr SeaError) Error() string {
	switch seaErr {
	case ErrPipe:
		return "Pipeline error"
	case ErrData:
		return "data error"
	case ErrFields:
		return "Fields error"
	case ErrGData:
		return "GData error"
	case ErrChData:
		return "ChData"
	case ErrModSpec:
		return "ModSpec error"
	case ErrNNModel:
		return "NNModel error"
	case ErrDiags:
		return "model diagnostics error"
	case ErrVecData:
		return "VecData error"
	}

	return "error"
}

func Wrapper(e error, text string) error {
	return fmt.Errorf("%v: %w", text, e)
}

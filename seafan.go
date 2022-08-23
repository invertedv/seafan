// Package seafan is a set of tools for model building using gorgonia (https://pkg.go.dev/gorgonia.org/gorgonia)
// as the model-build engine.
//
// Seafan features:
//
// - A data pipeline based on chutils (https://github.com/invertedv/chutils) to access files and ClickHouse.
//   - Point-and-shoot specification of the data
//   - Simple specification of one-hot features
//
// - A wrapper around gorgonia for neural nets that marries to the pipeline.
//   - Simple specification of models, including embeddings
//   - A fit method with optional early stopping and callbacks
//   - Methods for saving and loading models
//
// - Model diagnostics for categorical targets.
//   - KS plots
//   - Decile plots
//
// - Utilities.
//   - Plotting wrapper for plotly (https://github.com/MetalBlueberry/go-plotly) for the most-commonly used plots in model-building.
//   - Numeric struct for (x,y) data and plotting and descriptive statistics.
package seafan

// Verbose controls amount of printing
var Verbose = true

// Browser is the browser to use for plotting.
var Browser = "firefox"

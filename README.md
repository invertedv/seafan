## Seafan
[![Go Report Card](https://goreportcard.com/badge/github.com/invertedv/seafan)](https://goreportcard.com/report/github.com/invertedv/seafan)
[![godoc](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/mod/github.com/invertedv/seafan?tab=overview)

Package seafan is a set of tools for model building using [gorgonia](https://pkg.go.dev/gorgonia.org/gorgonia@v0.9.17)
as the model-build engine.

Seafan features:

- A data pipeline based on [chutils](https://github.com/invertedv/chutils) to access files and ClickHouse.
  - Point-and-shoot specification of the data
  - Simple specification of one-hot features

- A wrapper around gorgonia for neural nets that marries to the pipeline.
  - Simple specification of models, including embeddings
  - A fit method with optional early stopping and callbacks
  - Methods for saving and loading models

- Model diagnostics for categorical targets.
  - KS plots
  - Decile plots

- Utilities.
  - Plotting wrapper for [plotly](https://github.com/MetalBlueberry/go-plotly) for the most-commonly used plots in model-building.
  - Numeric struct for (x,y) data and plotting and descriptive statistics.

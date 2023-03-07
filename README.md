## Seafan
[![Go Report Card](https://goreportcard.com/badge/github.com/invertedv/seafan)](https://goreportcard.com/report/github.com/invertedv/seafan)
[![godoc](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/mod/github.com/invertedv/seafan?tab=overview)

Package seafan is a set of tools for building DNN models. The build engine is [gorgonia](https://pkg.go.dev/gorgonia.org/gorgonia@v0.9.17).

Seafan features:

- A data pipeline based on [chutils](https://github.com/invertedv/chutils) to access files and ClickHouse tables.
  - Point-and-shoot specification of the data
  - Simple specification of one-hot features
  - Functions of fields in the pipeline can be calculated from, and optionally added back,
     to the pipeline using the built-in expression parser (see Expr2Tree).
<br><br>
- A wrapper around gorgonia that meshes to the pipeline.
  - Simple specification of models, including embeddings
  - A fit method with optional early stopping
  - Callbacks during model fit
  - Saving and loading models
<br><br>
- Model diagnostics for categorical targets.
  - KS plots
  - Decile plots
  - Marginal effects plots
<br><br>
- Utilities.
  - Plotting wrapper for [plotly](https://github.com/MetalBlueberry/go-plotly) for xy plots.
  - Numeric struct for (x,y) data and plotting and descriptive statistics.

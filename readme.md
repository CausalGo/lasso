# LASSO Regression in Go

[![Go Reference](https://pkg.go.dev/badge/github.com/yourusername/lasso.svg)](https://pkg.go.dev/github.com/yourusername/lasso)
[![Tests](https://github.com/yourusername/lasso/actions/workflows/go.yml/badge.svg)](https://github.com/yourusername/lasso/actions)

Efficient LASSO (Least Absolute Shrinkage and Selection Operator) regression implementation using parallel coordinate descent.

## Features

- Parallel coordinate descent optimization
- Feature standardization
- Early stopping
- Training metrics tracking
- Comprehensive evaluation metrics (R², MSE, MAE)
- Configurable regularization strength

## Installation

```bash
go get github.com/yourusername/lasso
```

## Usage

```go
package main

import (
	"fmt"
	
	"github.com/yourusername/lasso"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Training data
	X := mat.NewDense(4, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
	y := []float64{3, 7, 11, 15}

	// Configure training
	cfg := lasso.NewDefaultConfig()
	cfg.Lambda = 0.1
	cfg.Verbose = true

	// Train model
	model := lasso.Fit(X, y, cfg)

	// Make predictions
	newX := mat.NewDense(2, 2, []float64{2, 3, 4, 5})
	predictions := model.Predict(newX)
	fmt.Println("Predictions:", predictions)
	
	// Evaluate model
	score := model.Score(X, y)
	fmt.Printf("R² score: %.4f\n", score)
}
```

## Documentation

Full documentation is available on [pkg.go.dev](https://pkg.go.dev/github.com/yourusername/lasso).

## Benchmarks

Benchmark results on synthetic data (10000 samples, 100 features):

```
BenchmarkFit-12          50     24823612 ns/op    53.72 MB/s
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

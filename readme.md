# 🎯 LASSO Regression in Go

[![Go Reference](https://pkg.go.dev/badge/github.com/CausalGo/lasso.svg)](https://pkg.go.dev/github.com/CausalGo/lasso)
[![Tests](https://github.com/CausalGo/lasso/actions/workflows/go.yml/badge.svg)](https://github.com/CausalGo/lasso/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/CausalGo/lasso)](https://goreportcard.com/report/github.com/CausalGo/lasso)

Efficient parallel implementation of LASSO (Least Absolute Shrinkage and Selection Operator) regression using coordinate descent optimization. Designed for performance and scalability with Go's concurrency model.

## Features ✨

- ⚡ **Parallel coordinate descent** - Leverages goroutines for concurrent feature updates
- 📉 **L1 regularization** - Automatic feature selection and model simplification
- 🎯 **Early stopping** - Terminates training when convergence is detected
- 📊 **Metrics tracking** - Records MSE, R², and weight deltas during training
- 🔧 **Feature standardization** - Automatic data preprocessing
- 📈 **Comprehensive evaluation** - Supports R², MSE, and MAE metrics
- 📝 **Training history** - Access detailed logs of each iteration
- ⚙️ **Configurable parameters** - Tune lambda, tolerance, and parallel jobs

## Installation 📦

```bash
go get github.com/CausalGo/lasso
```

## Quick Start 🚀

```go
package main

import (
	"fmt"
	
	"github.com/CausalGo/lasso"
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
	cfg.Lambda = 0.1      // Regularization strength
	cfg.Verbose = true    // Enable training logs
	cfg.NJobs = 2         // Use 2 parallel workers

	// Train model
	model := lasso.Fit(X, y, cfg)

	// Make predictions
	newX := mat.NewDense(2, 2, []float64{
		2, 3,
		4, 5,
	})
	predictions := model.Predict(newX)
	fmt.Println("Predictions:", predictions) // [5.0001, 9.0000]
	
	// Evaluate model
	score := model.Score(X, y)
	fmt.Printf("R² score: %.4f\n", score) // 1.0000
}
```

## Advanced Usage 🧠

### Custom Configuration
```go
cfg := &lasso.Config{
	Lambda:     0.05,    // Regularization parameter
	MaxIter:    2000,    // Maximum iterations
	Tol:        1e-5,    // Convergence tolerance
	NJobs:      4,       // Parallel workers
	Standardize: true,   // Standardize features
	Verbose:    true,    // Show training logs
	LogStep:    50,      // Log every 50 iterations
	EarlyStop:  true,    // Enable early stopping
	StopAfter:  15,      // Stop after 15 iterations without improvement
}
```

### Accessing Training History
```go
model := lasso.Fit(X, y, cfg)

// Analyze training progress
for _, log := range model.History {
	if log.Iteration%100 == 0 {
		fmt.Printf("Iter %d: MSE=%.4f R²=%.4f\n", 
			log.Iteration, log.MSE, log.R2)
	}
}
```

### Saving and Loading Models
```go
// Save model to JSON
err := model.Save("model.json")
if err != nil {
	panic(err)
}

// Load model from JSON
loadedModel, err := lasso.Load("model.json")
if err != nil {
	panic(err)
}
```

## Performance Benchmarks ⏱️

Benchmark results on synthetic dataset (10,000 samples, 100 features):

| Lambda | Iterations | Time (ms) | Active Features |
|--------|------------|-----------|-----------------|
| 0.01   | 142        | 245       | 73              |
| 0.1    | 78         | 132       | 45              |
| 1.0    | 35         | 58        | 12              |

```
BenchmarkFit-8   	      50	  24823612 ns/op
```

## Documentation 📚

Full documentation is available on [pkg.go.dev](https://pkg.go.dev/github.com/CausalGo/lasso)

## Contributing 🤝

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a pull request

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**CausalGo** - Machine learning tools for causal analysis in Go

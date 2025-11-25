# LASSO Regression in Go

> **Pure Go implementation of LASSO and Elastic Net regression** - sklearn-compatible API

[![GitHub Release](https://img.shields.io/github/v/release/CausalGo/lasso?include_prereleases&style=flat-square&logo=github&color=blue)](https://github.com/CausalGo/lasso/releases/latest)
[![Go Version](https://img.shields.io/badge/Go-1.25%2B-00ADD8?style=flat-square&logo=go)](https://go.dev/dl/)
[![Go Reference](https://pkg.go.dev/badge/github.com/CausalGo/lasso.svg)](https://pkg.go.dev/github.com/CausalGo/lasso)
[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/CausalGo/lasso/go.yml?branch=main&style=flat-square&logo=github-actions&label=CI)](https://github.com/CausalGo/lasso/actions)
[![Go Report Card](https://goreportcard.com/badge/github.com/CausalGo/lasso?style=flat-square)](https://goreportcard.com/report/github.com/CausalGo/lasso)
[![codecov](https://img.shields.io/codecov/c/github/CausalGo/lasso?style=flat-square&logo=codecov)](https://codecov.io/gh/CausalGo/lasso)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/CausalGo/lasso?style=flat-square&logo=github)](https://github.com/CausalGo/lasso/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/CausalGo/lasso?style=flat-square&logo=github)](https://github.com/CausalGo/lasso/issues)

---

Efficient implementation of LASSO (Least Absolute Shrinkage and Selection Operator) and Elastic Net regression using sequential coordinate descent optimization. Supports pure LASSO (L1), pure Ridge (L2), and balanced Elastic Net regularization. Optimized for cache locality and numerical stability.

## Features ✨

- ⚡ **Sequential coordinate descent** - Optimized for cache locality and performance
- 📉 **L1 regularization** - Automatic feature selection and model simplification
- 🔀 **Elastic Net support** - Combine L1 (LASSO) and L2 (Ridge) regularization
- 🎯 **Early stopping** - Terminates training when convergence is detected
- 📊 **Metrics tracking** - Records MSE, R², and weight deltas during training
- 🔧 **Feature standardization** - Automatic data preprocessing
- 📈 **Comprehensive evaluation** - Supports R², MSE, and MAE metrics
- 📝 **Training history** - Access detailed logs of each iteration
- ⚙️ **Configurable parameters** - Tune lambda, alpha, tolerance, and more
- 🔄 **Cross-validation** - K-fold CV for automatic lambda selection
- 💾 **Model persistence** - Save/Load models to JSON

## Requirements

- Go 1.25+

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

	// Train model
	model, err := lasso.Fit(X, y, cfg)
	if err != nil {
		panic(err)
	}

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
	Lambda:      0.05,    // Regularization parameter
	Alpha:       1.0,     // Elastic Net mixing: 1.0=LASSO, 0.0=Ridge, 0.5=balanced
	MaxIter:     2000,    // Maximum iterations
	Tol:         1e-5,    // Convergence tolerance
	Standardize: true,    // Standardize features
	Verbose:     true,    // Show training logs
	LogStep:     50,      // Log every 50 iterations
	EarlyStop:   true,    // Enable early stopping
	StopAfter:   15,      // Stop after 15 iterations without improvement
	MinDelta:    1e-5,    // Minimum improvement for early stopping
}
```

### Elastic Net Regularization
```go
// Pure LASSO (L1 only) - produces sparse solutions
cfg := lasso.NewDefaultConfig()
cfg.Lambda = 0.1
cfg.Alpha = 1.0  // Default: pure LASSO

// Elastic Net (L1 + L2 mix) - balanced regularization
cfg.Alpha = 0.5  // 50% L1, 50% L2

// Pure Ridge (L2 only) - no sparsity, all features active
cfg.Alpha = 0.0  // Pure Ridge regularization

model, err := lasso.Fit(X, y, cfg)
```

The Elastic Net objective function:
```
minimize: (1/2n) * ||y - Xw||² + λ * (α * ||w||₁ + (1-α) * ||w||²/2)
```

Where:
- `α = 1.0`: Pure LASSO - encourages sparsity (feature selection)
- `α = 0.0`: Pure Ridge - shrinks coefficients without sparsity
- `0 < α < 1`: Elastic Net - balances sparsity and stability

### Accessing Training History
```go
model, err := lasso.Fit(X, y, cfg)
if err != nil {
	panic(err)
}

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

### Cross-Validation for Lambda Selection
```go
// Automatic lambda selection via k-fold cross-validation
result, err := lasso.CrossValidate(X, y, &lasso.CVConfig{
	Lambdas: []float64{0.001, 0.01, 0.1, 1.0}, // Or nil for auto-generation
	NFolds:  5,                                  // 5-fold CV
	Scoring: "mse",                              // "mse", "r2", or "mae"
	Seed:    42,                                 // For reproducibility
	Config:  lasso.NewDefaultConfig(),           // Base training config
})
if err != nil {
	panic(err)
}

fmt.Printf("Best lambda: %.4f\n", result.BestLambda)
fmt.Printf("Best score: %.4f\n", result.BestScore)

// Use the best model (trained on full data with best lambda)
predictions := result.Model.Predict(newX)

// Access detailed CV results
for lambda, scores := range result.CVScores {
	fmt.Printf("Lambda %.4f: mean=%.4f, scores=%v\n",
		lambda, result.MeanScores[lambda], scores)
}
```

## Performance Benchmarks ⏱️

Run benchmarks locally:

```bash
go test -bench=. -run=^Benchmark -benchmem ./...
```

Key optimizations:
- **Sequential coordinate descent** - better cache locality than parallel
- **RawMatrix() access** - direct slice operations, no bounds checking overhead
- **Minimal allocations** - reuses buffers via `predictInto()`

## Documentation 📚

- [API Reference](https://pkg.go.dev/github.com/CausalGo/lasso) - Full documentation on pkg.go.dev
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide and Git workflow
- [CHANGELOG.md](CHANGELOG.md) - Release history
- [ROADMAP.md](ROADMAP.md) - Development roadmap
- [SECURITY.md](SECURITY.md) - Security policy

## Contributing 🤝

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on:

- Git-Flow branching model
- Commit message conventions
- Code quality standards
- Pull request requirements

Quick start:
```bash
git checkout develop
git checkout -b feature/amazing-feature
# Make changes...
go fmt ./... && golangci-lint run && go test -race ./...
git commit -m "feat: add amazing feature"
git push origin feature/amazing-feature
```

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**[CausalGo](https://github.com/CausalGo)** - Machine learning tools for causal analysis in Go

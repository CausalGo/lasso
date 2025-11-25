package main

import (
	"fmt"
	"math"

	"github.com/causalgo/lasso"
	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("=== Elastic Net Regression Example ===")

	// Create synthetic dataset with correlated features
	nSamples := 100
	nFeatures := 10
	X := mat.NewDense(nSamples, nFeatures, nil)
	y := make([]float64, nSamples)

	// Generate data: y = 3*x0 + 2*x1 + noise, rest features are noise
	fmt.Println("Generating synthetic dataset:")
	fmt.Printf("  Samples: %d, Features: %d\n", nSamples, nFeatures)
	fmt.Println("  True model: y = 3*x0 + 2*x1 + noise")
	fmt.Println()

	for i := 0; i < nSamples; i++ {
		X.Set(i, 0, float64(i)*0.1)
		X.Set(i, 1, float64(i)*0.05)
		for j := 2; j < nFeatures; j++ {
			X.Set(i, j, float64(i*j)*0.01)
		}
		y[i] = 3.0*X.At(i, 0) + 2.0*X.At(i, 1) + float64(i%5)*0.1
	}

	lambda := 0.1

	// Test 1: Pure LASSO (Alpha = 1.0)
	fmt.Println("--- Test 1: Pure LASSO (α=1.0) ---")
	runModel(X, y, lambda, 1.0, "LASSO")

	// Test 2: Balanced Elastic Net (Alpha = 0.5)
	fmt.Println("\n--- Test 2: Elastic Net (α=0.5) ---")
	runModel(X, y, lambda, 0.5, "Elastic Net")

	// Test 3: Pure Ridge (Alpha = 0.0)
	fmt.Println("\n--- Test 3: Pure Ridge (α=0.0) ---")
	runModel(X, y, lambda, 0.0, "Ridge")

	// Compare all three models
	fmt.Println("\n=== Model Comparison ===")
	fmt.Println("Alpha | Method       | R²     | Non-zero | L1 norm | L2 norm")
	fmt.Println("------|--------------|--------|----------|---------|--------")

	compareModel(X, y, lambda, 1.0, "LASSO")
	compareModel(X, y, lambda, 0.5, "Elastic Net")
	compareModel(X, y, lambda, 0.0, "Ridge")

	fmt.Println("\n=== Key Observations ===")
	fmt.Println("1. LASSO (α=1.0): Produces sparse solution (few non-zero weights)")
	fmt.Println("2. Ridge (α=0.0): Keeps all features with small weights")
	fmt.Println("3. Elastic Net (α=0.5): Balance between sparsity and regularization")
	fmt.Println("\nElastic Net is particularly useful when:")
	fmt.Println("  - Features are correlated")
	fmt.Println("  - You want some sparsity but also stability")
	fmt.Println("  - Ridge is too dense and LASSO is too aggressive")
}

func runModel(X *mat.Dense, y []float64, lambda, alpha float64, name string) {
	cfg := lasso.NewDefaultConfig()
	cfg.Lambda = lambda
	cfg.Alpha = alpha
	cfg.Verbose = false
	cfg.MaxIter = 1000

	model, err := lasso.Fit(X, y, cfg)
	if err != nil {
		fmt.Printf("Error fitting %s: %v\n", name, err)
		return
	}

	// Count non-zero weights
	nonZeroCount := 0
	l1Norm := 0.0
	l2Norm := 0.0
	for _, w := range model.Weights {
		if math.Abs(w) > 1e-6 {
			nonZeroCount++
		}
		l1Norm += math.Abs(w)
		l2Norm += w * w
	}
	l2Norm = math.Sqrt(l2Norm)

	r2 := model.Score(X, y)

	fmt.Printf("Config: λ=%.2f, α=%.1f\n", lambda, alpha)
	fmt.Printf("Results:\n")
	fmt.Printf("  R² Score: %.4f\n", r2)
	fmt.Printf("  Non-zero weights: %d/%d\n", nonZeroCount, len(model.Weights))
	fmt.Printf("  L1 norm (sparsity): %.4f\n", l1Norm)
	fmt.Printf("  L2 norm (magnitude): %.4f\n", l2Norm)
	fmt.Printf("  Iterations: %d\n", len(model.History))

	// Show top 3 features by absolute weight
	type weightInfo struct {
		index  int
		weight float64
	}
	weights := make([]weightInfo, 0)
	for i, w := range model.Weights {
		if math.Abs(w) > 1e-6 {
			weights = append(weights, weightInfo{i, w})
		}
	}

	fmt.Println("  Top features:")
	shown := 0
	for _, wi := range weights {
		if shown >= 3 {
			break
		}
		fmt.Printf("    Feature %d: %.4f\n", wi.index, wi.weight)
		shown++
	}
	if shown == 0 {
		fmt.Println("    (all weights near zero)")
	}
}

func compareModel(X *mat.Dense, y []float64, lambda, alpha float64, name string) {
	cfg := lasso.NewDefaultConfig()
	cfg.Lambda = lambda
	cfg.Alpha = alpha
	cfg.Verbose = false
	cfg.MaxIter = 1000

	model, err := lasso.Fit(X, y, cfg)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	nonZeroCount := 0
	l1Norm := 0.0
	l2Norm := 0.0
	for _, w := range model.Weights {
		if math.Abs(w) > 1e-6 {
			nonZeroCount++
		}
		l1Norm += math.Abs(w)
		l2Norm += w * w
	}
	l2Norm = math.Sqrt(l2Norm)

	r2 := model.Score(X, y)

	fmt.Printf("%-5.1f | %-12s | %6.4f | %8d | %7.4f | %7.4f\n",
		alpha, name, r2, nonZeroCount, l1Norm, l2Norm)
}

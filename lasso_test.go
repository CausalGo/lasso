package lasso

import (
	"gonum.org/v1/gonum/floats"
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLASSORegression(t *testing.T) {
	// Create synthetic dataset
	X := mat.NewDense(4, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
	y := []float64{3, 7, 11, 15}

	// Training configuration
	cfg := NewDefaultConfig()
	cfg.Lambda = 0.001
	cfg.Verbose = false
	cfg.MaxIter = 1000

	// Train model
	model := Fit(X, y, cfg)

	// Verify coefficients with tolerance
	tol := 1e-3
	expectedWeights := []float64{2.0, 0.0}
	for i, w := range model.Weights {
		if math.Abs(w-expectedWeights[i]) > tol {
			t.Errorf("Weight[%d] = %.6f, want %.1f ± %.3f", i, w, expectedWeights[i], tol)
		}
	}

	// Verify intercept with tolerance
	expectedIntercept := 1.0
	if math.Abs(model.Intercept-expectedIntercept) > tol {
		t.Errorf("Intercept = %.6f, want %.1f ± %.3f", model.Intercept, expectedIntercept, tol)
	}

	// Test predictions with tolerance
	predictions := model.Predict(X)
	expectedPredictions := []float64{3.0, 7.0, 11.0, 15.0}
	for i, pred := range predictions {
		if math.Abs(pred-expectedPredictions[i]) > tol {
			t.Errorf("Prediction[%d] = %.6f, want %.1f ± %.3f", i, pred, expectedPredictions[i], tol)
		}
	}

	// Test metrics
	score := model.Score(X, y)
	if math.Abs(score-1.0) > tol {
		t.Errorf("R² score = %.6f, want 1.0 ± %.3f", score, tol)
	}

	mse := model.MSE(X, y)
	if mse > tol {
		t.Errorf("MSE = %.6f, want < %.3f", mse, tol)
	}
}

func TestHighRegularization(t *testing.T) {
	X := mat.NewDense(4, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
	y := []float64{3, 7, 11, 15}
	meanY := floats.Sum(y) / float64(len(y))

	cfg := NewDefaultConfig()
	cfg.Lambda = 100.0 // Very high regularization
	cfg.Verbose = false

	model := Fit(X, y, cfg)

	// Verify all weights are near zero
	tol := 1e-5
	for i, w := range model.Weights {
		if math.Abs(w) > tol {
			t.Errorf("Weight[%d] = %.6f, want 0.0 ± %.5f", i, w, tol)
		}
	}

	// Verify intercept is near mean(y)
	if math.Abs(model.Intercept-meanY) > tol {
		t.Errorf("Intercept = %.6f, want %.6f ± %.5f", model.Intercept, meanY, tol)
	}
}

func TestStandardization(t *testing.T) {
	// Create dataset with different scales
	X := mat.NewDense(4, 2, []float64{
		1, 200,
		3, 400,
		5, 600,
		7, 800,
	})
	y := []float64{3, 7, 11, 15}

	cfg := NewDefaultConfig()
	cfg.Lambda = 0.1
	cfg.Standardize = true
	cfg.Verbose = false

	model := Fit(X, y, cfg)

	// Predictions should be reasonable
	predictions := model.Predict(X)
	for i, pred := range predictions {
		diff := math.Abs(pred - y[i])
		if diff > 1.0 {
			t.Errorf("Large prediction error: %.4f vs %.4f", pred, y[i])
		}
	}
}

func TestConvergence(t *testing.T) {
	X := mat.NewDense(100, 5, nil)
	y := make([]float64, 100)

	// Random data (in real test use proper randomization)
	for i := 0; i < 100; i++ {
		for j := 0; j < 5; j++ {
			X.Set(i, j, float64(i+j))
		}
		y[i] = float64(i)
	}

	cfg := NewDefaultConfig()
	cfg.Lambda = 0.1
	cfg.Tol = 1e-6
	cfg.EarlyStop = false
	cfg.Verbose = false

	model := Fit(X, y, cfg)

	// Should converge before max iterations
	if len(model.History) == cfg.MaxIter {
		t.Error("Model did not converge")
	}

	// Final maxDelta should be below tolerance
	lastIter := model.History[len(model.History)-1]
	if lastIter.MaxDelta > cfg.Tol {
		t.Errorf("MaxDelta %.2e > tolerance %.2e", lastIter.MaxDelta, cfg.Tol)
	}
}

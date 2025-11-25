package lasso

import (
	"gonum.org/v1/gonum/floats"
	"math"
	"os"
	"path/filepath"
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
	cfg.Lambda = 0.00025 // Scaled for 4 samples to match sklearn behavior
	cfg.Verbose = false
	cfg.MaxIter = 1000

	// Train model
	model, err := Fit(X, y, cfg)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

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
	cfg.Lambda = 25.0 // Very high regularization (scaled for 4 samples)
	cfg.Verbose = false

	model, err := Fit(X, y, cfg)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

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

	model, err := Fit(X, y, cfg)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

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

	model, err := Fit(X, y, cfg)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

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

// BenchmarkFit benchmarks the LASSO Fit function
func BenchmarkFit(b *testing.B) {
	// Create realistic dataset
	nSamples := 100
	nFeatures := 20
	X := mat.NewDense(nSamples, nFeatures, nil)
	y := make([]float64, nSamples)

	// Initialize with synthetic data
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, float64(i*nFeatures+j))
		}
		y[i] = float64(i)
	}

	cfg := NewDefaultConfig()
	cfg.Lambda = 0.1
	cfg.Verbose = false
	cfg.MaxIter = 100 // Fixed number of iterations for consistent benchmarking

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Fit(X, y, cfg)
	}
}

// BenchmarkFitLarge benchmarks with larger dataset
func BenchmarkFitLarge(b *testing.B) {
	nSamples := 500
	nFeatures := 50
	X := mat.NewDense(nSamples, nFeatures, nil)
	y := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, float64(i*nFeatures+j))
		}
		y[i] = float64(i)
	}

	cfg := NewDefaultConfig()
	cfg.Lambda = 0.1
	cfg.Verbose = false
	cfg.MaxIter = 100

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Fit(X, y, cfg)
	}
}

// BenchmarkPredict benchmarks the Predict function
func BenchmarkPredict(b *testing.B) {
	nSamples := 1000
	nFeatures := 50
	X := mat.NewDense(nSamples, nFeatures, nil)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, float64(i*nFeatures+j))
		}
	}

	model := &LassoModel{
		Weights:   make([]float64, nFeatures),
		Intercept: 1.0,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = model.Predict(X)
	}
}

// TestSaveLoad tests model serialization and deserialization
func TestSaveLoad(t *testing.T) {
	// Create temporary directory for test files
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "model.json")

	// Create and train a model
	X := mat.NewDense(4, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
	y := []float64{3, 7, 11, 15}

	cfg := NewDefaultConfig()
	cfg.Lambda = 0.00025
	cfg.Verbose = false
	cfg.MaxIter = 1000

	originalModel, err := Fit(X, y, cfg)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Save the model
	err = originalModel.Save(modelPath)
	if err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	// Verify file was created
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Fatalf("Model file was not created at %s", modelPath)
	}

	// Load the model
	loadedModel, err := Load(modelPath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// Verify loaded model matches original model
	tol := 1e-10

	// Check weights
	if len(loadedModel.Weights) != len(originalModel.Weights) {
		t.Fatalf("Loaded weights length %d != original %d", len(loadedModel.Weights), len(originalModel.Weights))
	}
	for i, w := range loadedModel.Weights {
		if math.Abs(w-originalModel.Weights[i]) > tol {
			t.Errorf("Weight[%d]: loaded %.10f != original %.10f", i, w, originalModel.Weights[i])
		}
	}

	// Check intercept
	if math.Abs(loadedModel.Intercept-originalModel.Intercept) > tol {
		t.Errorf("Intercept: loaded %.10f != original %.10f", loadedModel.Intercept, originalModel.Intercept)
	}

	// Check lambda
	if math.Abs(loadedModel.Lambda-originalModel.Lambda) > tol {
		t.Errorf("Lambda: loaded %.10f != original %.10f", loadedModel.Lambda, originalModel.Lambda)
	}

	// Check history length
	if len(loadedModel.History) != len(originalModel.History) {
		t.Fatalf("History length: loaded %d != original %d", len(loadedModel.History), len(originalModel.History))
	}

	// Check a few history entries
	if len(originalModel.History) > 0 {
		lastIdx := len(originalModel.History) - 1
		if loadedModel.History[lastIdx].Iteration != originalModel.History[lastIdx].Iteration {
			t.Errorf("Last iteration: loaded %d != original %d",
				loadedModel.History[lastIdx].Iteration, originalModel.History[lastIdx].Iteration)
		}
		if math.Abs(loadedModel.History[lastIdx].MSE-originalModel.History[lastIdx].MSE) > tol {
			t.Errorf("Last MSE: loaded %.10f != original %.10f",
				loadedModel.History[lastIdx].MSE, originalModel.History[lastIdx].MSE)
		}
	}

	// Verify loaded model makes same predictions
	originalPred := originalModel.Predict(X)
	loadedPred := loadedModel.Predict(X)

	if len(originalPred) != len(loadedPred) {
		t.Fatalf("Predictions length mismatch: %d != %d", len(loadedPred), len(originalPred))
	}

	for i := range originalPred {
		if math.Abs(loadedPred[i]-originalPred[i]) > tol {
			t.Errorf("Prediction[%d]: loaded %.10f != original %.10f", i, loadedPred[i], originalPred[i])
		}
	}

	// Verify metrics are identical
	originalScore := originalModel.Score(X, y)
	loadedScore := loadedModel.Score(X, y)
	if math.Abs(loadedScore-originalScore) > tol {
		t.Errorf("R² score: loaded %.10f != original %.10f", loadedScore, originalScore)
	}
}

// TestFitValidationErrors tests that Fit returns errors for invalid input
func TestFitValidationErrors(t *testing.T) {
	cfg := NewDefaultConfig()
	cfg.Verbose = false

	t.Run("mismatched dimensions", func(t *testing.T) {
		X := mat.NewDense(4, 2, []float64{1, 2, 3, 4, 5, 6, 7, 8})
		y := []float64{1, 2, 3} // Wrong length

		_, err := Fit(X, y, cfg)
		if err == nil {
			t.Error("Expected error for mismatched dimensions, got nil")
		}
	})

	t.Run("NaN in y", func(t *testing.T) {
		X := mat.NewDense(4, 2, []float64{1, 2, 3, 4, 5, 6, 7, 8})
		y := []float64{1, 2, math.NaN(), 4}

		_, err := Fit(X, y, cfg)
		if err == nil {
			t.Error("Expected error for NaN in y, got nil")
		}
	})

	t.Run("Inf in y", func(t *testing.T) {
		X := mat.NewDense(4, 2, []float64{1, 2, 3, 4, 5, 6, 7, 8})
		y := []float64{1, 2, math.Inf(1), 4}

		_, err := Fit(X, y, cfg)
		if err == nil {
			t.Error("Expected error for Inf in y, got nil")
		}
	})

	t.Run("NaN in X", func(t *testing.T) {
		X := mat.NewDense(4, 2, []float64{1, 2, 3, math.NaN(), 5, 6, 7, 8})
		y := []float64{1, 2, 3, 4}

		_, err := Fit(X, y, cfg)
		if err == nil {
			t.Error("Expected error for NaN in X, got nil")
		}
	})

	t.Run("Inf in X", func(t *testing.T) {
		X := mat.NewDense(4, 2, []float64{1, 2, 3, 4, math.Inf(-1), 6, 7, 8})
		y := []float64{1, 2, 3, 4}

		_, err := Fit(X, y, cfg)
		if err == nil {
			t.Error("Expected error for Inf in X, got nil")
		}
	})
}

// TestSaveLoadErrors tests error handling in Save/Load
func TestSaveLoadErrors(t *testing.T) {
	// Test Save to invalid path
	model := &LassoModel{
		Weights:   []float64{1.0, 2.0},
		Intercept: 0.5,
		Lambda:    0.1,
	}

	invalidPath := filepath.Join("nonexistent_directory_xyz", "model.json")
	err := model.Save(invalidPath)
	if err == nil {
		t.Error("Expected error when saving to invalid path, got nil")
	}

	// Test Load from nonexistent file
	_, err = Load("nonexistent_model_xyz.json")
	if err == nil {
		t.Error("Expected error when loading nonexistent file, got nil")
	}

	// Test Load from invalid JSON
	tempDir := t.TempDir()
	invalidJSONPath := filepath.Join(tempDir, "invalid.json")
	err = os.WriteFile(invalidJSONPath, []byte("{invalid json}"), 0644)
	if err != nil {
		t.Fatalf("Failed to create invalid JSON file: %v", err)
	}

	_, err = Load(invalidJSONPath)
	if err == nil {
		t.Error("Expected error when loading invalid JSON, got nil")
	}
}

// TestElasticNet tests Elastic Net regularization with different alpha values
func TestElasticNet(t *testing.T) {
	// Create synthetic dataset with correlated features
	X := mat.NewDense(100, 10, nil)
	y := make([]float64, 100)

	// Generate data: y = 3*x0 + 2*x1 + noise, rest features are noise
	for i := 0; i < 100; i++ {
		X.Set(i, 0, float64(i)*0.1)
		X.Set(i, 1, float64(i)*0.05)
		for j := 2; j < 10; j++ {
			X.Set(i, j, float64(i*j)*0.01)
		}
		y[i] = 3.0*X.At(i, 0) + 2.0*X.At(i, 1) + float64(i%5)*0.1
	}

	lambda := 0.1
	tol := 0.1 // Looser tolerance for comparison

	// Test 1: Alpha = 1.0 (Pure LASSO)
	t.Run("Alpha=1.0 Pure LASSO", func(t *testing.T) {
		cfg := NewDefaultConfig()
		cfg.Lambda = lambda
		cfg.Alpha = 1.0 // Pure LASSO
		cfg.Verbose = false
		cfg.MaxIter = 1000

		model, err := Fit(X, y, cfg)
		if err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		// LASSO should produce sparse solution
		nonZeroCount := 0
		for _, w := range model.Weights {
			if math.Abs(w) > 1e-6 {
				nonZeroCount++
			}
		}

		// LASSO should select only few features
		if nonZeroCount > 5 {
			t.Logf("LASSO selected %d features (expected sparse solution)", nonZeroCount)
		}

		// Save LASSO R² for comparison
		lassoR2 := model.Score(X, y)
		t.Logf("LASSO (α=1.0): R²=%.4f, Non-zero weights=%d", lassoR2, nonZeroCount)
	})

	// Test 2: Alpha = 0.0 (Pure Ridge)
	t.Run("Alpha=0.0 Pure Ridge", func(t *testing.T) {
		cfg := NewDefaultConfig()
		cfg.Lambda = lambda
		cfg.Alpha = 0.0 // Pure Ridge (L2 only)
		cfg.Verbose = false
		cfg.MaxIter = 1000

		model, err := Fit(X, y, cfg)
		if err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		// Ridge should NOT produce sparse solution
		nonZeroCount := 0
		for _, w := range model.Weights {
			if math.Abs(w) > 1e-6 {
				nonZeroCount++
			}
		}

		// Ridge should keep most/all features
		if nonZeroCount < 8 {
			t.Errorf("Ridge (α=0.0) produced sparse solution: %d non-zero weights, expected most features active", nonZeroCount)
		}

		ridgeR2 := model.Score(X, y)
		t.Logf("Ridge (α=0.0): R²=%.4f, Non-zero weights=%d", ridgeR2, nonZeroCount)
	})

	// Test 3: Alpha = 0.5 (Balanced Elastic Net)
	t.Run("Alpha=0.5 Balanced Elastic Net", func(t *testing.T) {
		cfg := NewDefaultConfig()
		cfg.Lambda = lambda
		cfg.Alpha = 0.5 // Balanced mix
		cfg.Verbose = false
		cfg.MaxIter = 1000

		model, err := Fit(X, y, cfg)
		if err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		// Elastic Net should be between LASSO and Ridge in sparsity
		nonZeroCount := 0
		for _, w := range model.Weights {
			if math.Abs(w) > 1e-6 {
				nonZeroCount++
			}
		}

		elasticR2 := model.Score(X, y)
		t.Logf("Elastic Net (α=0.5): R²=%.4f, Non-zero weights=%d", elasticR2, nonZeroCount)

		// Elastic Net should have intermediate sparsity
		if nonZeroCount < 3 || nonZeroCount > 9 {
			t.Logf("Warning: Elastic Net sparsity (%d) may be outside expected range [3, 9]", nonZeroCount)
		}
	})

	// Test 4: Verify Alpha=1.0 matches original LASSO behavior
	t.Run("Alpha=1.0 matches LASSO", func(t *testing.T) {
		// Simple test case from TestLASSORegression
		XSimple := mat.NewDense(4, 2, []float64{
			1, 2,
			3, 4,
			5, 6,
			7, 8,
		})
		ySimple := []float64{3, 7, 11, 15}

		cfgLasso := NewDefaultConfig()
		cfgLasso.Lambda = 0.00025
		cfgLasso.Alpha = 1.0 // Explicit LASSO
		cfgLasso.Verbose = false
		cfgLasso.MaxIter = 1000

		modelLasso, err := Fit(XSimple, ySimple, cfgLasso)
		if err != nil {
			t.Fatalf("Fit with Alpha=1.0 failed: %v", err)
		}

		// Should match expected LASSO behavior
		expectedWeights := []float64{2.0, 0.0}
		for i, w := range modelLasso.Weights {
			if math.Abs(w-expectedWeights[i]) > tol {
				t.Errorf("Weight[%d] with Alpha=1.0: %.6f, want %.1f ± %.1f", i, w, expectedWeights[i], tol)
			}
		}

		expectedIntercept := 1.0
		if math.Abs(modelLasso.Intercept-expectedIntercept) > tol {
			t.Errorf("Intercept with Alpha=1.0: %.6f, want %.1f ± %.1f", modelLasso.Intercept, expectedIntercept, tol)
		}

		t.Logf("Alpha=1.0 successfully replicates LASSO behavior")
	})

	// Test 5: Verify different Alpha values produce different results
	t.Run("Different Alpha values produce different models", func(t *testing.T) {
		alphas := []float64{0.0, 0.5, 1.0}
		models := make([]*LassoModel, len(alphas))

		for i, alpha := range alphas {
			cfg := NewDefaultConfig()
			cfg.Lambda = lambda
			cfg.Alpha = alpha
			cfg.Verbose = false
			cfg.MaxIter = 1000

			model, err := Fit(X, y, cfg)
			if err != nil {
				t.Fatalf("Fit with Alpha=%.1f failed: %v", alpha, err)
			}
			models[i] = model
		}

		// Verify models are different (weights should differ)
		for i := 0; i < len(models)-1; i++ {
			weightsDiffer := false
			for j := range models[i].Weights {
				if math.Abs(models[i].Weights[j]-models[i+1].Weights[j]) > 1e-4 {
					weightsDiffer = true
					break
				}
			}
			if !weightsDiffer {
				t.Errorf("Models with Alpha=%.1f and Alpha=%.1f produced identical weights (expected different)",
					alphas[i], alphas[i+1])
			}
		}

		t.Logf("Different Alpha values successfully produce distinct models")
	})
}

// TestCrossValidate tests k-fold cross-validation
func TestCrossValidate(t *testing.T) {
	// Create simple linear dataset: y = 2*x1 + x2 + 1
	X := mat.NewDense(30, 2, nil)
	y := make([]float64, 30)

	for i := 0; i < 30; i++ {
		x1 := float64(i) * 0.1
		x2 := float64(i) * 0.2
		X.Set(i, 0, x1)
		X.Set(i, 1, x2)
		y[i] = 2.0*x1 + x2 + 1.0 + float64(i%3)*0.1 // Small noise
	}

	t.Run("basic cross-validation with 3 folds", func(t *testing.T) {
		baseCfg := NewDefaultConfig()
		baseCfg.MaxIter = 500
		baseCfg.Verbose = false

		cvCfg := &CVConfig{
			Lambdas: []float64{0.001, 0.01, 0.1, 1.0},
			NFolds:  3,
			Seed:    42,
			Scoring: "mse",
			Config:  baseCfg,
		}

		result, err := CrossValidate(X, y, cvCfg)
		if err != nil {
			t.Fatalf("CrossValidate failed: %v", err)
		}

		// Verify result structure
		if result == nil {
			t.Fatal("Expected non-nil result")
		}
		if result.Model == nil {
			t.Fatal("Expected trained model in result")
		}

		// Verify best lambda is one of the provided lambdas
		found := false
		for _, lambda := range cvCfg.Lambdas {
			if math.Abs(result.BestLambda-lambda) < 1e-10 {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("BestLambda %.6f not in provided lambdas %v", result.BestLambda, cvCfg.Lambdas)
		}

		// Verify CVScores contains all lambdas
		if len(result.CVScores) != len(cvCfg.Lambdas) {
			t.Errorf("CVScores length %d != lambdas length %d", len(result.CVScores), len(cvCfg.Lambdas))
		}

		// Verify each lambda has correct number of fold scores
		for lambda, scores := range result.CVScores {
			if len(scores) != cvCfg.NFolds {
				t.Errorf("Lambda %.6f has %d fold scores, expected %d", lambda, len(scores), cvCfg.NFolds)
			}
		}

		// Verify MeanScores matches CVScores
		for lambda, scores := range result.CVScores {
			expectedMean := floats.Sum(scores) / float64(len(scores))
			actualMean := result.MeanScores[lambda]
			if math.Abs(expectedMean-actualMean) > 1e-10 {
				t.Errorf("Mean score for lambda %.6f: got %.6f, want %.6f", lambda, actualMean, expectedMean)
			}
		}

		// Verify model is trained with best lambda
		if math.Abs(result.Model.Lambda-result.BestLambda) > 1e-10 {
			t.Errorf("Model lambda %.6f != best lambda %.6f", result.Model.Lambda, result.BestLambda)
		}

		// Verify model makes reasonable predictions
		predictions := result.Model.Predict(X)
		if len(predictions) != len(y) {
			t.Fatalf("Predictions length %d != y length %d", len(predictions), len(y))
		}

		// Check MSE is reasonable
		mse := result.Model.MSE(X, y)
		if mse > 1.0 {
			t.Errorf("Model MSE %.4f seems too high for this simple dataset", mse)
		}

		t.Logf("Best lambda: %.6f, Best score (MSE): %.6f", result.BestLambda, result.BestScore)
		t.Logf("Final model R²: %.4f, MSE: %.4f", result.Model.Score(X, y), mse)
	})

	t.Run("cross-validation with R² scoring", func(t *testing.T) {
		baseCfg := NewDefaultConfig()
		baseCfg.MaxIter = 500
		baseCfg.Verbose = false

		cvCfg := &CVConfig{
			Lambdas: []float64{0.001, 0.01, 0.1},
			NFolds:  3,
			Seed:    42,
			Scoring: "r2", // R² scoring
			Config:  baseCfg,
		}

		result, err := CrossValidate(X, y, cvCfg)
		if err != nil {
			t.Fatalf("CrossValidate with R² scoring failed: %v", err)
		}

		// With R², best score should be highest (closer to 1)
		// Verify it's reasonable
		if result.BestScore < 0.5 {
			t.Errorf("Best R² score %.4f seems too low", result.BestScore)
		}

		t.Logf("Best lambda: %.6f, Best score (R²): %.4f", result.BestLambda, result.BestScore)
	})

	t.Run("cross-validation with MAE scoring", func(t *testing.T) {
		baseCfg := NewDefaultConfig()
		baseCfg.MaxIter = 500
		baseCfg.Verbose = false

		cvCfg := &CVConfig{
			Lambdas: []float64{0.001, 0.01, 0.1},
			NFolds:  3,
			Seed:    42,
			Scoring: "mae", // MAE scoring
			Config:  baseCfg,
		}

		result, err := CrossValidate(X, y, cvCfg)
		if err != nil {
			t.Fatalf("CrossValidate with MAE scoring failed: %v", err)
		}

		// With MAE, lower is better
		if result.BestScore > 1.0 {
			t.Errorf("Best MAE score %.4f seems too high", result.BestScore)
		}

		t.Logf("Best lambda: %.6f, Best score (MAE): %.4f", result.BestLambda, result.BestScore)
	})

	t.Run("cross-validation with auto-generated lambdas", func(t *testing.T) {
		baseCfg := NewDefaultConfig()
		baseCfg.MaxIter = 500
		baseCfg.Verbose = false

		cvCfg := &CVConfig{
			Lambdas: nil, // Auto-generate
			NFolds:  5,
			Seed:    42,
			Scoring: "mse",
			Config:  baseCfg,
		}

		result, err := CrossValidate(X, y, cvCfg)
		if err != nil {
			t.Fatalf("CrossValidate with auto-generated lambdas failed: %v", err)
		}

		// Should have generated 20 lambdas (default in generateLambdaPath)
		if len(result.CVScores) != 20 {
			t.Errorf("Expected 20 auto-generated lambdas, got %d", len(result.CVScores))
		}

		t.Logf("Auto-generated %d lambdas, best: %.6f", len(result.CVScores), result.BestLambda)
	})
}

// TestCrossValidateEdgeCases tests edge cases and error handling
func TestCrossValidateEdgeCases(t *testing.T) {
	X := mat.NewDense(10, 2, nil)
	y := make([]float64, 10)
	for i := 0; i < 10; i++ {
		X.Set(i, 0, float64(i))
		X.Set(i, 1, float64(i)*2)
		y[i] = float64(i)
	}

	baseCfg := NewDefaultConfig()
	baseCfg.Verbose = false

	t.Run("mismatched X and y dimensions", func(t *testing.T) {
		yWrong := []float64{1, 2, 3} // Wrong length

		cvCfg := &CVConfig{
			Lambdas: []float64{0.1},
			NFolds:  3,
			Config:  baseCfg,
		}

		_, err := CrossValidate(X, yWrong, cvCfg)
		if err == nil {
			t.Error("Expected error for mismatched dimensions, got nil")
		}
	})

	t.Run("invalid scoring metric", func(t *testing.T) {
		cvCfg := &CVConfig{
			Lambdas: []float64{0.1},
			NFolds:  3,
			Scoring: "invalid_metric",
			Config:  baseCfg,
		}

		_, err := CrossValidate(X, y, cvCfg)
		if err == nil {
			t.Error("Expected error for invalid scoring metric, got nil")
		}
	})

	t.Run("too many folds", func(t *testing.T) {
		cvCfg := &CVConfig{
			Lambdas: []float64{0.1},
			NFolds:  100, // More than samples
			Config:  baseCfg,
		}

		_, err := CrossValidate(X, y, cvCfg)
		if err == nil {
			t.Error("Expected error for too many folds, got nil")
		}
	})

	t.Run("empty lambda list", func(t *testing.T) {
		cvCfg := &CVConfig{
			Lambdas: []float64{}, // Empty
			NFolds:  3,
			Config:  baseCfg,
		}

		_, err := CrossValidate(X, y, cvCfg)
		if err == nil {
			t.Error("Expected error for empty lambda list, got nil")
		}
	})

	t.Run("default NFolds when not specified", func(t *testing.T) {
		cvCfg := &CVConfig{
			Lambdas: []float64{0.1},
			NFolds:  0, // Should default to 5
			Config:  baseCfg,
		}

		result, err := CrossValidate(X, y, cvCfg)
		if err != nil {
			t.Fatalf("CrossValidate with default NFolds failed: %v", err)
		}

		// Should use 5 folds by default
		for _, scores := range result.CVScores {
			if len(scores) != 5 {
				t.Errorf("Expected 5 folds (default), got %d", len(scores))
			}
		}
	})

	t.Run("NaN in input data", func(t *testing.T) {
		XBad := mat.NewDense(10, 2, nil)
		for i := 0; i < 10; i++ {
			XBad.Set(i, 0, float64(i))
			XBad.Set(i, 1, math.NaN()) // NaN value
		}

		cvCfg := &CVConfig{
			Lambdas: []float64{0.1},
			NFolds:  3,
			Config:  baseCfg,
		}

		_, err := CrossValidate(XBad, y, cvCfg)
		if err == nil {
			t.Error("Expected error for NaN in X, got nil")
		}
	})
}

// TestExtractSubset tests the extractSubset helper function
func TestExtractSubset(t *testing.T) {
	// Create test data
	X := mat.NewDense(5, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		13, 14, 15,
	})
	y := []float64{1, 2, 3, 4, 5}

	t.Run("extract subset with specific indices", func(t *testing.T) {
		indices := []int{0, 2, 4}
		subX, subY := extractSubset(X, y, indices)

		// Verify dimensions
		r, c := subX.Dims()
		if r != len(indices) {
			t.Errorf("Subset rows: got %d, want %d", r, len(indices))
		}
		if c != 3 {
			t.Errorf("Subset cols: got %d, want 3", c)
		}
		if len(subY) != len(indices) {
			t.Errorf("Subset y length: got %d, want %d", len(subY), len(indices))
		}

		// Verify values
		expectedX := [][]float64{
			{1, 2, 3},
			{7, 8, 9},
			{13, 14, 15},
		}
		expectedY := []float64{1, 3, 5}

		for i := 0; i < len(indices); i++ {
			for j := 0; j < 3; j++ {
				got := subX.At(i, j)
				want := expectedX[i][j]
				if got != want {
					t.Errorf("subX[%d,%d]: got %.0f, want %.0f", i, j, got, want)
				}
			}
			if subY[i] != expectedY[i] {
				t.Errorf("subY[%d]: got %.0f, want %.0f", i, subY[i], expectedY[i])
			}
		}
	})

	t.Run("extract all indices", func(t *testing.T) {
		indices := []int{0, 1, 2, 3, 4}
		subX, subY := extractSubset(X, y, indices)

		r, c := subX.Dims()
		if r != 5 || c != 3 {
			t.Errorf("Full subset dimensions: got (%d,%d), want (5,3)", r, c)
		}

		// Should match original data
		for i := 0; i < 5; i++ {
			for j := 0; j < 3; j++ {
				if subX.At(i, j) != X.At(i, j) {
					t.Errorf("Mismatch at [%d,%d]", i, j)
				}
			}
			if subY[i] != y[i] {
				t.Errorf("Mismatch in y at [%d]", i)
			}
		}
	})

	t.Run("extract single index", func(t *testing.T) {
		indices := []int{2}
		subX, subY := extractSubset(X, y, indices)

		r, c := subX.Dims()
		if r != 1 || c != 3 {
			t.Errorf("Single subset dimensions: got (%d,%d), want (1,3)", r, c)
		}

		// Verify it's the correct row
		for j := 0; j < 3; j++ {
			want := X.At(2, j)
			got := subX.At(0, j)
			if got != want {
				t.Errorf("subX[0,%d]: got %.0f, want %.0f", j, got, want)
			}
		}
		if subY[0] != y[2] {
			t.Errorf("subY[0]: got %.0f, want %.0f", subY[0], y[2])
		}
	})
}

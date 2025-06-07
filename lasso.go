// Package lasso implements LASSO (Least Absolute Shrinkage and Selection Operator) regression
// with parallel coordinate descent optimization.
package lasso

import (
	"fmt"
	"math"
	"sync"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// LassoModel represents a trained LASSO regression model.
type LassoModel struct {
	Weights   []float64      // Regression coefficients
	Intercept float64        // Bias term
	Lambda    float64        // Regularization parameter
	History   []IterationLog // Training history
}

// IterationLog contains training metrics for a single iteration.
type IterationLog struct {
	Iteration int
	Timestamp time.Time
	MaxDelta  float64 // Maximum weight change
	MSE       float64 // Mean Squared Error
	R2        float64 // R-squared coefficient
}

// Config holds training parameters for LASSO regression.
type Config struct {
	Lambda      float64 // Regularization strength (λ)
	MaxIter     int     // Maximum number of iterations
	Tol         float64 // Convergence tolerance
	NJobs       int     // Number of parallel workers
	Standardize bool    // Standardize features
	Verbose     bool    // Enable training logs
	LogStep     int     // Logging frequency
	EarlyStop   bool    // Enable early stopping
	StopAfter   int     // Stop after N iterations without improvement
	MinDelta    float64 // Minimum improvement for early stopping
}

// NewDefaultConfig returns recommended default parameters.
func NewDefaultConfig() *Config {
	return &Config{
		Lambda:      0.01,
		MaxIter:     1000,
		Tol:         1e-4,
		NJobs:       4,
		Standardize: true,
		Verbose:     false,
		LogStep:     10,
		EarlyStop:   true,
		StopAfter:   20,
		MinDelta:    1e-5,
	}
}

// Fit trains a LASSO regression model using coordinate descent.
func Fit(X *mat.Dense, y []float64, cfg *Config) *LassoModel {
	startTime := time.Now()
	nSamples, nFeatures := X.Dims()

	if len(y) != nSamples {
		panic("X and y have different number of samples")
	}

	// Create working copies
	XData := mat.DenseCopyOf(X)
	yData := make([]float64, len(y))
	copy(yData, y)

	// Standardize features and target
	var xMeans, xStds []float64
	yMean := 0.0
	if cfg.Standardize {
		xMeans, xStds = standardizeFeatures(XData)
		yMean = centerTarget(yData)
	}

	// Initialize model parameters
	weights := make([]float64, nFeatures)
	intercept := 0.0
	activeSet := make([]bool, nFeatures) // Active feature tracking
	residuals := make([]float64, nSamples)
	copy(residuals, yData)

	// Training history tracking
	history := []IterationLog{}
	bestMSE := math.MaxFloat64
	noImproveCount := 0

	if cfg.Verbose {
		fmt.Println("Starting LASSO training")
		fmt.Printf("Params: λ=%.4f, MaxIter=%d, Tol=%.0e\n", cfg.Lambda, cfg.MaxIter, cfg.Tol)
		fmt.Printf("Samples: %d, Features: %d\n", nSamples, nFeatures)
	}

	// Main training loop
	for iter := 0; iter < cfg.MaxIter; iter++ {
		iterStart := time.Now()
		maxDelta := 0.0
		prevWeights := make([]float64, nFeatures)
		copy(prevWeights, weights)

		// Parallel coordinate update
		var wg sync.WaitGroup
		ch := make(chan int, nFeatures)

		for i := 0; i < cfg.NJobs; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := range ch {
					if !activeSet[j] && iter > 0 {
						continue // Skip inactive features
					}

					oldWeight := weights[j]
					if oldWeight != 0 {
						updateResiduals(XData, residuals, j, oldWeight)
					}

					// Compute correlation and norm
					rho, xtx := 0.0, 0.0
					for i := 0; i < nSamples; i++ {
						xVal := XData.At(i, j)
						rho += xVal * residuals[i]
						xtx += xVal * xVal
					}

					// Apply soft-thresholding
					newWeight := softThreshold(rho, cfg.Lambda) / (xtx + 1e-8)
					delta := math.Abs(newWeight - oldWeight)
					if delta > maxDelta {
						maxDelta = delta
					}

					// Update weights and residuals
					if newWeight != 0 {
						activeSet[j] = true
						updateResiduals(XData, residuals, j, -newWeight)
						weights[j] = newWeight
					} else {
						weights[j] = 0
					}
				}
			}()
		}

		// Distribute feature indices
		for j := 0; j < nFeatures; j++ {
			ch <- j
		}
		close(ch)
		wg.Wait()

		// Update intercept
		meanResidual := floats.Sum(residuals) / float64(nSamples)
		newIntercept := intercept + meanResidual
		deltaIntercept := math.Abs(newIntercept - intercept)
		intercept = newIntercept
		floats.AddConst(-meanResidual, residuals)

		// Compute metrics
		predictions := predict(XData, weights, intercept)
		mse := meanSquaredError(yData, predictions)
		r2 := rSquared(yData, predictions)

		// Record history
		logEntry := IterationLog{
			Iteration: iter,
			Timestamp: time.Now(),
			MaxDelta:  maxDelta,
			MSE:       mse,
			R2:        r2,
		}
		history = append(history, logEntry)

		// Log progress
		if cfg.Verbose && (iter%cfg.LogStep == 0 || iter == cfg.MaxIter-1) {
			duration := time.Since(iterStart)
			activeCount := countActive(activeSet)

			fmt.Printf("Iter %4d: MSE=%.4f R²=%.4f |Δ|=%.2e |Δb|=%.2e | Active=%d/%d | Time=%s\n",
				iter, mse, r2, maxDelta, deltaIntercept, activeCount, nFeatures, duration.Round(time.Microsecond))
		}

		// Check convergence
		if maxDelta < cfg.Tol {
			if cfg.Verbose {
				fmt.Printf("Converged at iteration %d: |Δ| < %.0e\n", iter, cfg.Tol)
			}
			break
		}

		// Early stopping
		if cfg.EarlyStop {
			if mse < bestMSE-cfg.MinDelta {
				bestMSE = mse
				noImproveCount = 0
			} else {
				noImproveCount++
			}

			if noImproveCount >= cfg.StopAfter {
				if cfg.Verbose {
					fmt.Printf("Early stopping at iteration %d: no improvement for %d iterations\n",
						iter, noImproveCount)
				}
				break
			}
		}
	}

	// Reverse standardization transformations
	if cfg.Standardize {
		denormalizeWeights(weights, xMeans, xStds)
		intercept = denormalizeIntercept(intercept, weights, xMeans, xStds, yMean)
	}

	// Finalize model
	model := &LassoModel{
		Weights:   weights,
		Intercept: intercept,
		Lambda:    cfg.Lambda,
		History:   history,
	}

	if cfg.Verbose {
		totalDuration := time.Since(startTime)
		fmt.Printf("\nTraining completed in %s\n", totalDuration.Round(time.Millisecond))
		fmt.Printf("Weights: %v\n", weights)
		fmt.Printf("Intercept: %.4f\n", intercept)
	}

	return model
}

// Predict returns predictions for input samples.
func (m *LassoModel) Predict(X *mat.Dense) []float64 {
	nSamples, nFeatures := X.Dims()
	predictions := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		sum := m.Intercept
		for j := 0; j < nFeatures; j++ {
			sum += X.At(i, j) * m.Weights[j]
		}
		predictions[i] = sum
	}
	return predictions
}

// Score returns the R² score for given data.
func (m *LassoModel) Score(X *mat.Dense, y []float64) float64 {
	pred := m.Predict(X)
	return rSquared(y, pred)
}

// MSE returns the mean squared error for given data.
func (m *LassoModel) MSE(X *mat.Dense, y []float64) float64 {
	pred := m.Predict(X)
	return meanSquaredError(y, pred)
}

// MAE returns the mean absolute error for given data.
func (m *LassoModel) MAE(X *mat.Dense, y []float64) float64 {
	pred := m.Predict(X)
	return meanAbsoluteError(y, pred)
}

// --- Helper Functions ---

// predict makes predictions without model overhead
func predict(X *mat.Dense, weights []float64, intercept float64) []float64 {
	nSamples, nFeatures := X.Dims()
	pred := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		sum := intercept
		for j := 0; j < nFeatures; j++ {
			sum += X.At(i, j) * weights[j]
		}
		pred[i] = sum
	}
	return pred
}

// standardizeFeatures centers and scales features (in-place)
func standardizeFeatures(X *mat.Dense) (means, stds []float64) {
	nSamples, nFeatures := X.Dims()
	means = make([]float64, nFeatures)
	stds = make([]float64, nFeatures)

	for j := 0; j < nFeatures; j++ {
		col := make([]float64, nSamples)
		for i := 0; i < nSamples; i++ {
			col[i] = X.At(i, j)
		}

		// Compute mean and standard deviation
		means[j] = floats.Sum(col) / float64(nSamples)
		variance := 0.0
		for i := range col {
			centered := col[i] - means[j]
			col[i] = centered
			variance += centered * centered
		}

		stds[j] = math.Sqrt(variance / float64(nSamples-1))
		if stds[j] < 1e-8 {
			stds[j] = 1.0
		} else {
			for i := range col {
				col[i] /= stds[j]
			}
		}

		X.SetCol(j, col)
	}
	return means, stds
}

// centerTarget centers the target variable (in-place)
func centerTarget(y []float64) float64 {
	mean := floats.Sum(y) / float64(len(y))
	floats.AddConst(-mean, y)
	return mean
}

// updateResiduals updates residuals when a weight changes
func updateResiduals(X *mat.Dense, residuals []float64, j int, delta float64) {
	nSamples, _ := X.Dims()
	for i := 0; i < nSamples; i++ {
		residuals[i] += delta * X.At(i, j)
	}
}

// softThreshold applies the soft-thresholding operator
func softThreshold(z, lambda float64) float64 {
	if z > lambda {
		return z - lambda
	} else if z < -lambda {
		return z + lambda
	}
	return 0
}

// denormalizeWeights converts weights to original feature scale
func denormalizeWeights(weights []float64, means, stds []float64) {
	for j := range weights {
		if stds[j] != 0 {
			weights[j] /= stds[j]
		}
	}
}

// denormalizeIntercept converts intercept to original scale
func denormalizeIntercept(intercept float64, weights, means, stds []float64, yMean float64) float64 {
	dot := 0.0
	for j := range weights {
		dot += means[j] * weights[j]
	}
	return yMean + intercept - dot
}

// countActive counts active features
func countActive(activeSet []bool) int {
	count := 0
	for _, active := range activeSet {
		if active {
			count++
		}
	}
	return count
}

// --- Evaluation Metrics ---

// meanSquaredError calculates MSE
func meanSquaredError(yTrue, yPred []float64) float64 {
	if len(yTrue) != len(yPred) {
		panic("input lengths must match")
	}
	sum := 0.0
	for i := range yTrue {
		diff := yTrue[i] - yPred[i]
		sum += diff * diff
	}
	return sum / float64(len(yTrue))
}

// rSquared calculates coefficient of determination
func rSquared(yTrue, yPred []float64) float64 {
	if len(yTrue) != len(yPred) {
		panic("input lengths must match")
	}
	mean := floats.Sum(yTrue) / float64(len(yTrue))

	tss := 0.0 // Total sum of squares
	rss := 0.0 // Residual sum of squares
	for i := range yTrue {
		tss += (yTrue[i] - mean) * (yTrue[i] - mean)
		diff := yTrue[i] - yPred[i]
		rss += diff * diff
	}

	if tss < 1e-15 {
		return 1
	}
	return 1 - rss/tss
}

// meanAbsoluteError calculates MAE
func meanAbsoluteError(yTrue, yPred []float64) float64 {
	if len(yTrue) != len(yPred) {
		panic("input lengths must match")
	}
	sum := 0.0
	for i := range yTrue {
		sum += math.Abs(yTrue[i] - yPred[i])
	}
	return sum / float64(len(yTrue))
}

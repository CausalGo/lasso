// Package lasso implements LASSO (Least Absolute Shrinkage and Selection Operator) regression
// with sequential coordinate descent optimization. This implementation prioritizes cache locality
// and numerical stability over parallelism for small-to-medium datasets.
package lasso

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
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
	Alpha       float64 // Elastic Net mixing parameter: 1.0 = LASSO (L1), 0.0 = Ridge (L2), 0.5 = balanced mix
	MaxIter     int     // Maximum number of iterations
	Tol         float64 // Convergence tolerance
	NJobs       int     // Deprecated: kept for API compatibility, not used in sequential implementation
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
		Alpha:       1.0, // Pure LASSO by default (backward compatible)
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

// validateInput checks for NaN/Inf values in input data
func validateInput(X *mat.Dense, y []float64) error {
	nSamples, nFeatures := X.Dims()

	// Check y for NaN/Inf
	for i, val := range y {
		if math.IsNaN(val) {
			return fmt.Errorf("y contains NaN at index %d", i)
		}
		if math.IsInf(val, 0) {
			return fmt.Errorf("y contains Inf at index %d", i)
		}
	}

	// Check X for NaN/Inf
	rawX := X.RawMatrix()
	xData := rawX.Data
	stride := rawX.Stride

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			val := xData[i*stride+j]
			if math.IsNaN(val) {
				return fmt.Errorf("x contains NaN at position (%d, %d)", i, j)
			}
			if math.IsInf(val, 0) {
				return fmt.Errorf("x contains Inf at position (%d, %d)", i, j)
			}
		}
	}

	return nil
}

// Fit trains a LASSO regression model using sequential coordinate descent.
// Returns (*LassoModel, error) where error is non-nil if validation fails.
func Fit(X *mat.Dense, y []float64, cfg *Config) (*LassoModel, error) {
	startTime := time.Now()
	nSamples, nFeatures := X.Dims()

	if len(y) != nSamples {
		return nil, fmt.Errorf("x and y have different number of samples: x has %d samples, y has %d", nSamples, len(y))
	}

	// Validate input data for NaN/Inf
	if err := validateInput(X, y); err != nil {
		return nil, fmt.Errorf("input validation failed: %w", err)
	}

	// Create working copies to avoid modifying original data
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
		fmt.Printf("Params: λ=%.4f, α=%.2f, MaxIter=%d, Tol=%.0e\n", cfg.Lambda, cfg.Alpha, cfg.MaxIter, cfg.Tol)
		fmt.Printf("Samples: %d, Features: %d\n", nSamples, nFeatures)
	}

	// Get raw matrix for optimized access
	rawX := XData.RawMatrix()
	xData := rawX.Data
	stride := rawX.Stride

	// Pre-allocate predictions slice to avoid per-iteration allocations
	predictions := make([]float64, nSamples)

	// Main training loop - sequential coordinate descent
	for iter := 0; iter < cfg.MaxIter; iter++ {
		iterStart := time.Now()
		maxDelta := 0.0

		// Update each feature sequentially
		for j := 0; j < nFeatures; j++ {
			// Skip inactive features after first iteration (active set optimization)
			if iter > 0 && !activeSet[j] {
				continue
			}

			oldWeight := weights[j]

			// Temporarily remove feature's contribution to residuals
			if oldWeight != 0 {
				updateResidualsOptimized(xData, stride, residuals, j, oldWeight)
			}

			// Compute correlation (X_j^T * residuals) and norm (||X_j||^2)
			// Using direct slice access for performance
			rho, xtx := 0.0, 0.0
			for i := 0; i < nSamples; i++ {
				xVal := xData[i*stride+j]
				rho += xVal * residuals[i]
				xtx += xVal * xVal
			}

			// Apply Elastic Net regularization
			// Objective: minimize (1/2n)||y - Xw||² + λ*(α*||w||₁ + (1-α)*||w||²/2)
			// Scale lambda by nSamples for sklearn compatibility
			l1Penalty := cfg.Alpha * cfg.Lambda * float64(nSamples)
			l2Penalty := (1 - cfg.Alpha) * cfg.Lambda * float64(nSamples)
			newWeight := softThreshold(rho, l1Penalty) / (xtx + l2Penalty + 1e-8)
			delta := math.Abs(newWeight - oldWeight)

			// Update residuals with new weight
			if newWeight != 0 {
				updateResidualsOptimized(xData, stride, residuals, j, -newWeight)
				activeSet[j] = true
			} else if oldWeight != 0 {
				// Feature becomes inactive
				activeSet[j] = false
			}

			// Update weight and track maximum change
			weights[j] = newWeight
			if delta > maxDelta {
				maxDelta = delta
			}
		}

		// Update intercept
		meanResidual := floats.Sum(residuals) / float64(nSamples)
		newIntercept := intercept + meanResidual
		deltaIntercept := math.Abs(newIntercept - intercept)
		intercept = newIntercept
		floats.AddConst(-meanResidual, residuals)

		// Compute performance metrics (reuse pre-allocated predictions slice)
		predictInto(XData, weights, intercept, predictions)
		mse := meanSquaredError(yData, predictions)
		r2 := rSquared(yData, predictions)

		// Record training history
		logEntry := IterationLog{
			Iteration: iter,
			Timestamp: time.Now(),
			MaxDelta:  maxDelta,
			MSE:       mse,
			R2:        r2,
		}
		history = append(history, logEntry)

		// Log progress if enabled
		if cfg.Verbose && (iter%cfg.LogStep == 0 || iter == cfg.MaxIter-1) {
			duration := time.Since(iterStart)
			activeCount := countActive(activeSet)

			fmt.Printf("Iter %4d: MSE=%.4f R²=%.4f |Δ|=%.2e |Δb|=%.2e | Active=%d/%d | Time=%s\n",
				iter, mse, r2, maxDelta, deltaIntercept, activeCount, nFeatures, duration.Round(time.Microsecond))
		}

		// Check convergence criterion
		if maxDelta < cfg.Tol {
			if cfg.Verbose {
				fmt.Printf("Converged at iteration %d: |Δ| < %.0e\n", iter, cfg.Tol)
			}
			break
		}

		// Early stopping based on MSE improvement
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

	// Finalize and return model
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

	return model, nil
}

// Predict returns predictions for input samples.
func (m *LassoModel) Predict(X *mat.Dense) []float64 {
	nSamples, nFeatures := X.Dims()
	predictions := make([]float64, nSamples)

	// Use direct slice access for better performance
	rawX := X.RawMatrix()
	xData := rawX.Data
	stride := rawX.Stride

	for i := 0; i < nSamples; i++ {
		sum := m.Intercept
		rowOffset := i * stride
		for j := 0; j < nFeatures; j++ {
			sum += xData[rowOffset+j] * m.Weights[j]
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

// Save serializes the model to JSON and writes it to a file.
// The model is saved with all its parameters (weights, intercept, lambda, history).
func (m *LassoModel) Save(filepath string) error {
	// Marshal model to JSON with indentation for readability
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal model: %w", err)
	}

	// Write to file with appropriate permissions
	if err := os.WriteFile(filepath, data, 0644); err != nil {
		return fmt.Errorf("failed to write model file: %w", err)
	}

	return nil
}

// Load deserializes a model from a JSON file.
// Returns the loaded model or an error if the file cannot be read or parsed.
func Load(filepath string) (*LassoModel, error) {
	// Read file contents
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read model file: %w", err)
	}

	// Unmarshal JSON into model
	var model LassoModel
	if err := json.Unmarshal(data, &model); err != nil {
		return nil, fmt.Errorf("failed to unmarshal model: %w", err)
	}

	return &model, nil
}

// --- Helper Functions ---

// predictInto makes predictions and writes them to pre-allocated slice
func predictInto(X *mat.Dense, weights []float64, intercept float64, pred []float64) {
	nSamples, nFeatures := X.Dims()

	// Use direct slice access for better performance
	rawX := X.RawMatrix()
	xData := rawX.Data
	stride := rawX.Stride

	for i := 0; i < nSamples; i++ {
		sum := intercept
		rowOffset := i * stride
		for j := 0; j < nFeatures; j++ {
			sum += xData[rowOffset+j] * weights[j]
		}
		pred[i] = sum
	}
}

// standardizeFeatures centers and scales features (in-place)
func standardizeFeatures(X *mat.Dense) (means, stds []float64) {
	nSamples, nFeatures := X.Dims()
	means = make([]float64, nFeatures)
	stds = make([]float64, nFeatures)

	// Get raw matrix for optimized access
	rawX := X.RawMatrix()
	xData := rawX.Data
	stride := rawX.Stride

	for j := 0; j < nFeatures; j++ {
		// Compute mean using direct access
		sum := 0.0
		for i := 0; i < nSamples; i++ {
			sum += xData[i*stride+j]
		}
		means[j] = sum / float64(nSamples)

		// Compute variance and center
		variance := 0.0
		for i := 0; i < nSamples; i++ {
			idx := i*stride + j
			centered := xData[idx] - means[j]
			xData[idx] = centered
			variance += centered * centered
		}

		// Compute std and scale (population variance for sklearn consistency)
		stds[j] = math.Sqrt(variance / float64(nSamples))
		if stds[j] < 1e-8 {
			stds[j] = 1.0
		} else {
			for i := 0; i < nSamples; i++ {
				idx := i*stride + j
				xData[idx] /= stds[j]
			}
		}
	}
	return means, stds
}

// centerTarget centers the target variable (in-place)
func centerTarget(y []float64) float64 {
	mean := floats.Sum(y) / float64(len(y))
	floats.AddConst(-mean, y)
	return mean
}

// updateResidualsOptimized updates residuals using direct slice access
func updateResidualsOptimized(xData []float64, stride int, residuals []float64, j int, delta float64) {
	for i := 0; i < len(residuals); i++ {
		residuals[i] += delta * xData[i*stride+j]
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

// --- Cross-Validation Types and Functions ---

// CVConfig holds cross-validation parameters.
type CVConfig struct {
	Lambdas []float64 // Lambda values to try (nil = auto-generate)
	NFolds  int       // Number of folds (default 5)
	Seed    int64     // Random seed for reproducibility
	Scoring string    // "mse", "r2", or "mae" (default "mse")
	Config  *Config   // Base training config
}

// CVResult holds cross-validation results.
type CVResult struct {
	BestLambda float64
	BestScore  float64
	Model      *LassoModel           // Best model trained on full data
	CVScores   map[float64][]float64 // Lambda -> scores per fold
	MeanScores map[float64]float64   // Lambda -> mean score
}

// generateLambdaPath generates a logarithmically-spaced sequence of lambda values.
// lambdaMax = max(|X'y|) / n, then logarithmic sequence down to lambdaMax/100.
func generateLambdaPath(X *mat.Dense, y []float64, nLambdas int) []float64 {
	nSamples, nFeatures := X.Dims()

	// Compute X'y (correlation between each feature and target)
	rawX := X.RawMatrix()
	xData := rawX.Data
	stride := rawX.Stride

	maxCorr := 0.0
	for j := 0; j < nFeatures; j++ {
		corr := 0.0
		for i := 0; i < nSamples; i++ {
			corr += xData[i*stride+j] * y[i]
		}
		absCorr := math.Abs(corr)
		if absCorr > maxCorr {
			maxCorr = absCorr
		}
	}

	// lambdaMax = max(|X'y|) / n (normalized by sample size)
	lambdaMax := maxCorr / float64(nSamples)
	lambdaMin := lambdaMax / 100.0 // Two orders of magnitude smaller

	// Generate logarithmically-spaced sequence
	lambdas := make([]float64, nLambdas)
	logMax := math.Log10(lambdaMax)
	logMin := math.Log10(lambdaMin)
	step := (logMax - logMin) / float64(nLambdas-1)

	for i := 0; i < nLambdas; i++ {
		lambdas[i] = math.Pow(10, logMax-float64(i)*step)
	}

	return lambdas
}

// kFoldSplit generates k-fold cross-validation indices.
// Returns a slice of k folds, where each fold is a slice of indices.
func kFoldSplit(n, nFolds int, seed int64) [][]int {
	// Create shuffled indices
	indices := make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = i
	}

	// Shuffle using deterministic seed
	rng := newRNG(seed)
	for i := n - 1; i > 0; i-- {
		j := rng.intn(i + 1)
		indices[i], indices[j] = indices[j], indices[i]
	}

	// Split into folds
	folds := make([][]int, nFolds)
	foldSize := n / nFolds
	remainder := n % nFolds

	start := 0
	for i := 0; i < nFolds; i++ {
		size := foldSize
		if i < remainder {
			size++ // Distribute remainder evenly
		}
		end := start + size
		folds[i] = make([]int, size)
		copy(folds[i], indices[start:end])
		start = end
	}

	return folds
}

// simpleRNG is a simple linear congruential generator for reproducible shuffling.
type simpleRNG struct {
	state uint64
}

func newRNG(seed int64) *simpleRNG {
	return &simpleRNG{state: uint64(seed)}
}

// intn returns a pseudo-random number in [0, n).
func (r *simpleRNG) intn(n int) int {
	// Linear congruential generator parameters (from Numerical Recipes)
	const (
		a = 1664525
		c = 1013904223
		m = 1 << 32
	)
	r.state = (a*r.state + c) % m
	return int(r.state % uint64(n))
}

// extractSubset extracts rows from X and corresponding values from y based on indices.
// Returns a new matrix and slice with the selected data.
func extractSubset(X *mat.Dense, y []float64, indices []int) (*mat.Dense, []float64) {
	_, nFeatures := X.Dims()
	nSamples := len(indices)

	// Create new matrix for subset
	subset := mat.NewDense(nSamples, nFeatures, nil)
	ySubset := make([]float64, nSamples)

	// Extract rows using optimized access
	rawX := X.RawMatrix()
	xData := rawX.Data
	stride := rawX.Stride

	for i, idx := range indices {
		// Copy row from X
		for j := 0; j < nFeatures; j++ {
			subset.Set(i, j, xData[idx*stride+j])
		}
		// Copy corresponding y value
		ySubset[i] = y[idx]
	}

	return subset, ySubset
}

// CrossValidate performs k-fold cross-validation to find the best lambda value.
// It trains models for each lambda on k-1 folds and validates on the remaining fold.
// Returns the best model trained on the full dataset with the optimal lambda.
func CrossValidate(X *mat.Dense, y []float64, cvCfg *CVConfig) (*CVResult, error) {
	nSamples, _ := X.Dims()

	// Validate inputs
	if len(y) != nSamples {
		return nil, fmt.Errorf("x and y have different number of samples: x has %d, y has %d", nSamples, len(y))
	}
	if err := validateInput(X, y); err != nil {
		return nil, fmt.Errorf("input validation failed: %w", err)
	}

	// Set defaults
	if cvCfg.NFolds <= 0 {
		cvCfg.NFolds = 5
	}
	if cvCfg.NFolds > nSamples {
		return nil, fmt.Errorf("nFolds (%d) cannot exceed number of samples (%d)", cvCfg.NFolds, nSamples)
	}
	if cvCfg.Scoring == "" {
		cvCfg.Scoring = "mse"
	}
	if cvCfg.Scoring != "mse" && cvCfg.Scoring != "r2" && cvCfg.Scoring != "mae" {
		return nil, fmt.Errorf("invalid scoring metric: %s (must be 'mse', 'r2', or 'mae')", cvCfg.Scoring)
	}
	if cvCfg.Config == nil {
		cvCfg.Config = NewDefaultConfig()
	}

	// Generate lambda path if not provided
	lambdas := cvCfg.Lambdas
	if lambdas == nil {
		lambdas = generateLambdaPath(X, y, 20)
	}
	if len(lambdas) == 0 {
		return nil, fmt.Errorf("no lambda values provided")
	}

	// Generate k-fold splits
	folds := kFoldSplit(nSamples, cvCfg.NFolds, cvCfg.Seed)

	// Initialize result tracking
	cvScores := make(map[float64][]float64)
	meanScores := make(map[float64]float64)

	// Suppress verbose output during CV
	baseCfg := *cvCfg.Config
	baseCfg.Verbose = false

	// For each lambda value
	for _, lambda := range lambdas {
		foldScores := make([]float64, cvCfg.NFolds)

		// For each fold
		for foldIdx, testIndices := range folds {
			// Create train indices (all folds except current)
			trainIndices := make([]int, 0, nSamples-len(testIndices))
			for i, fold := range folds {
				if i != foldIdx {
					trainIndices = append(trainIndices, fold...)
				}
			}

			// Extract train and test subsets
			XTrain, yTrain := extractSubset(X, y, trainIndices)
			XTest, yTest := extractSubset(X, y, testIndices)

			// Train model on train data
			trainCfg := baseCfg
			trainCfg.Lambda = lambda
			model, err := Fit(XTrain, yTrain, &trainCfg)
			if err != nil {
				return nil, fmt.Errorf("failed to fit model for lambda=%.6f, fold=%d: %w", lambda, foldIdx, err)
			}

			// Score on test data
			var score float64
			switch cvCfg.Scoring {
			case "mse":
				score = model.MSE(XTest, yTest)
			case "mae":
				score = model.MAE(XTest, yTest)
			case "r2":
				score = model.Score(XTest, yTest)
			}

			foldScores[foldIdx] = score
		}

		// Store scores for this lambda
		cvScores[lambda] = foldScores
		meanScores[lambda] = floats.Sum(foldScores) / float64(len(foldScores))
	}

	// Find best lambda based on scoring metric
	bestLambda := lambdas[0]
	bestScore := meanScores[bestLambda]

	for _, lambda := range lambdas[1:] {
		score := meanScores[lambda]
		isBetter := false

		switch cvCfg.Scoring {
		case "mse", "mae":
			// Lower is better for error metrics
			isBetter = score < bestScore
		case "r2":
			// Higher is better for R²
			isBetter = score > bestScore
		}

		if isBetter {
			bestLambda = lambda
			bestScore = score
		}
	}

	// Train final model on full dataset with best lambda
	finalCfg := *cvCfg.Config
	finalCfg.Lambda = bestLambda
	finalModel, err := Fit(X, y, &finalCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to fit final model with best lambda=%.6f: %w", bestLambda, err)
	}

	// Return results
	result := &CVResult{
		BestLambda: bestLambda,
		BestScore:  bestScore,
		Model:      finalModel,
		CVScores:   cvScores,
		MeanScores: meanScores,
	}

	return result, nil
}

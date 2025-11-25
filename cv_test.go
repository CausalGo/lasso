package lasso

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestGenerateLambdaPath tests lambda path generation
func TestGenerateLambdaPath(t *testing.T) {
	// Create simple dataset
	X := mat.NewDense(10, 3, []float64{
		1, 2, 3,
		2, 3, 4,
		3, 4, 5,
		4, 5, 6,
		5, 6, 7,
		6, 7, 8,
		7, 8, 9,
		8, 9, 10,
		9, 10, 11,
		10, 11, 12,
	})
	y := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

	lambdas := generateLambdaPath(X, y, 10)

	// Check number of lambdas
	if len(lambdas) != 10 {
		t.Errorf("Expected 10 lambdas, got %d", len(lambdas))
	}

	// Check that lambdas are sorted in descending order
	for i := 1; i < len(lambdas); i++ {
		if lambdas[i] > lambdas[i-1] {
			t.Errorf("Lambdas not in descending order at index %d: %.6f > %.6f",
				i, lambdas[i], lambdas[i-1])
		}
	}

	// Check that all lambdas are positive
	for i, lambda := range lambdas {
		if lambda <= 0 {
			t.Errorf("Lambda at index %d is not positive: %.6f", i, lambda)
		}
	}

	// Check that ratio between max and min is approximately 100
	ratio := lambdas[0] / lambdas[len(lambdas)-1]
	expectedRatio := 100.0
	if math.Abs(ratio-expectedRatio) > 1.0 {
		t.Errorf("Expected ratio ~%.1f, got %.2f", expectedRatio, ratio)
	}
}

// TestKFoldSplit tests k-fold splitting
func TestKFoldSplit(t *testing.T) {
	tests := []struct {
		n      int
		nFolds int
		seed   int64
	}{
		{100, 5, 42},
		{99, 5, 42},   // Test with uneven split
		{50, 10, 123}, // More folds
		{10, 2, 456},  // Minimal folds
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			folds := kFoldSplit(tt.n, tt.nFolds, tt.seed)

			// Check number of folds
			if len(folds) != tt.nFolds {
				t.Errorf("Expected %d folds, got %d", tt.nFolds, len(folds))
			}

			// Check that total number of samples is correct
			totalSamples := 0
			for _, fold := range folds {
				totalSamples += len(fold)
			}
			if totalSamples != tt.n {
				t.Errorf("Expected %d total samples, got %d", tt.n, totalSamples)
			}

			// Check that all indices are unique
			seen := make(map[int]bool)
			for _, fold := range folds {
				for _, idx := range fold {
					if seen[idx] {
						t.Errorf("Duplicate index %d found", idx)
					}
					if idx < 0 || idx >= tt.n {
						t.Errorf("Invalid index %d (n=%d)", idx, tt.n)
					}
					seen[idx] = true
				}
			}

			// Check that all indices from 0 to n-1 are present
			if len(seen) != tt.n {
				t.Errorf("Expected %d unique indices, got %d", tt.n, len(seen))
			}

			// Check reproducibility
			folds2 := kFoldSplit(tt.n, tt.nFolds, tt.seed)
			for i := range folds {
				if len(folds[i]) != len(folds2[i]) {
					t.Errorf("Fold sizes differ in reproducibility test")
				}
				for j := range folds[i] {
					if folds[i][j] != folds2[i][j] {
						t.Errorf("Fold content differs in reproducibility test")
						break
					}
				}
			}
		})
	}
}

// TestKFoldSplitDifferentSeeds tests that different seeds produce different splits
func TestKFoldSplitDifferentSeeds(t *testing.T) {
	n := 100
	nFolds := 5

	folds1 := kFoldSplit(n, nFolds, 42)
	folds2 := kFoldSplit(n, nFolds, 123)

	// At least one fold should be different
	different := false
	for i := range folds1 {
		if len(folds1[i]) != len(folds2[i]) {
			different = true
			break
		}
		for j := range folds1[i] {
			if folds1[i][j] != folds2[i][j] {
				different = true
				break
			}
		}
		if different {
			break
		}
	}

	if !different {
		t.Error("Expected different folds with different seeds")
	}
}

// TestSimpleRNG tests the random number generator
func TestSimpleRNG(t *testing.T) {
	rng := newRNG(42)

	// Generate some random numbers
	nums := make([]int, 100)
	for i := range nums {
		nums[i] = rng.intn(100)
		if nums[i] < 0 || nums[i] >= 100 {
			t.Errorf("Random number out of range: %d", nums[i])
		}
	}

	// Check reproducibility
	rng2 := newRNG(42)
	for i := range nums {
		num2 := rng2.intn(100)
		if nums[i] != num2 {
			t.Errorf("RNG not reproducible at index %d: %d != %d", i, nums[i], num2)
		}
	}

	// Check that different seeds produce different sequences
	rng3 := newRNG(123)
	different := false
	for i := 0; i < 100; i++ {
		if rng3.intn(100) != nums[i] {
			different = true
			break
		}
	}
	if !different {
		t.Error("Different seeds should produce different sequences")
	}
}

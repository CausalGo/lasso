package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"lasso"
)

func main() {
	X := mat.NewDense(4, 2, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	y := []float64{3, 7, 11, 15}

	cfg := lasso.NewDefaultConfig()
	cfg.Lambda = 0.001 // Уменьшенная регуляризация
	cfg.Verbose = true

	model := lasso.Fit(X, y, cfg)

	fmt.Println("\nУлучшенные веса:", model.Weights)
	fmt.Println("Улучшенное смещение:", model.Intercept)

	// Проверка точности
	for i := 0; i < 4; i++ {
		pred := model.Weights[0]*X.At(i, 0) + model.Weights[1]*X.At(i, 1) + model.Intercept
		fmt.Printf("X: [%.1f, %.1f] => y_true: %.1f, y_pred: %.4f\n",
			X.At(i, 0), X.At(i, 1), y[i], pred)
	}
}

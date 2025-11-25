package main

import (
	"fmt"
	"os"

	"github.com/causalgo/lasso"
	"gonum.org/v1/gonum/mat"
)

func main() {
	X := mat.NewDense(4, 2, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	y := []float64{3, 7, 11, 15}

	cfg := lasso.NewDefaultConfig()
	cfg.Lambda = 0.001 // Уменьшенная регуляризация
	cfg.Verbose = true

	model, err := lasso.Fit(X, y, cfg)
	if err != nil {
		fmt.Printf("Ошибка обучения: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\nУлучшенные веса:", model.Weights)
	fmt.Println("Улучшенное смещение:", model.Intercept)

	// Проверка точности
	fmt.Println("\nПрогнозы:")
	for i := 0; i < 4; i++ {
		pred := model.Weights[0]*X.At(i, 0) + model.Weights[1]*X.At(i, 1) + model.Intercept
		fmt.Printf("X: [%.1f, %.1f] => y_true: %.1f, y_pred: %.4f\n",
			X.At(i, 0), X.At(i, 1), y[i], pred)
	}

	// Сохранение модели
	modelPath := "lasso_model.json"
	fmt.Printf("\nСохранение модели в %s...\n", modelPath)
	if err := model.Save(modelPath); err != nil {
		fmt.Printf("Ошибка сохранения: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Модель успешно сохранена!")

	// Загрузка модели
	fmt.Printf("\nЗагрузка модели из %s...\n", modelPath)
	loadedModel, err := lasso.Load(modelPath)
	if err != nil {
		fmt.Printf("Ошибка загрузки: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Модель успешно загружена!")

	// Проверка, что загруженная модель работает
	fmt.Println("\nПрогнозы загруженной модели:")
	predictions := loadedModel.Predict(X)
	for i, pred := range predictions {
		fmt.Printf("X: [%.1f, %.1f] => y_true: %.1f, y_pred: %.4f\n",
			X.At(i, 0), X.At(i, 1), y[i], pred)
	}

	// Очистка
	fmt.Printf("\nУдаление тестового файла %s...\n", modelPath)
	os.Remove(modelPath)
}

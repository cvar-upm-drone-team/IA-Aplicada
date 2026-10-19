# Ejercicio 1 — Clasificación con K-Nearest Neighbors (KNN)

Clasificar imágenes de 6 categorías con un clasificador **KNN**, y medir
el efecto de reducir la dimensionalidad con PCA.

## Dataset

Descarga `cifar_subset.mat` de la [release del repositorio](https://github.com/GPatiA2/MUIA-IA_Aplicada/releases/tag/dataset-release-v1)
y colócalo en `data/cifar_subset.mat`. Formato completo y uso en Google
Colab: [`docs/dataset.md`](../docs/dataset.md).

```matlab
load('data/cifar_subset.mat');
% X_train 30000x32x32x3 uint8, y_train 30000x1 (etiquetas 1..6)
% X_test   6000x32x32x3 uint8, y_test   6000x1, class_names cell 1x6
```

## Tareas

### 1. Preprocesado

Aplana cada imagen de `32×32×3` a un vector de 3072 elementos y
normaliza a `[0, 1]`. En color: no conviertas a escala de grises.

Deberías acabar con `X_train` de `30000×3072` y `X_test` de `6000×3072`.

### 2. Clasificador KNN base

```matlab
mdl = fitcknn(X_train, y_train, 'NumNeighbors', 5);
[preds, scores] = predict(mdl, X_test);
```

### 3. Evaluación

`utils/evaluar.m` calcula accuracy, matriz de confusión,
precision/recall/F1 (por clase y macro-promediadas), AUC y AP
One-vs-Rest. Definiciones en [`docs/metrics.md`](../docs/metrics.md).

```matlab
addpath('utils');
evaluar(y_test, preds, scores, class_names);
```

### 4. Barrido de `k`

Repite el entrenamiento y el cálculo del accuracy de test para
`k = 1, 3, 5, 7, 9, 15, 25`. Representa el accuracy frente a `k` con
`plot` e identifica el mejor valor.

### 5. PCA antes de KNN

Proyecta los datos a 50 componentes principales y repite el
entrenamiento y la evaluación con el mejor `k` del apartado anterior.

```matlab
[coeff, scoreTrain, ~, ~, ~, mu] = pca(X_train, 'NumComponents', 50, 'Algorithm', 'eig');
scoreTest = (X_test - mu) * coeff;
```

Recoge en una tabla el accuracy con PCA y sin PCA.

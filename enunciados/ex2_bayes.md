# Ejercicio 2 — Clasificación Bayesiana (Naive Bayes Gaussiano)

Clasificar imágenes de 6 categorías con un **Naive Bayes Gaussiano**,
medir la correlación entre píxeles que su suposición de independencia
ignora, y medir el efecto de PCA.

## Dataset

Descarga `cifar_subset.mat` de la [release del repositorio](https://github.com/cvar-upm-drone-team/IA-Aplicada/releases/tag/dataset_release)
y colócalo en `data/cifar_subset.mat`. Formato completo y uso en Google
Colab: [`docs/dataset.md`](../docs/dataset.md).

```matlab
load('data/cifar_subset.mat');
% X_train 12000x32x32x3 uint8, y_train 12000x1 (etiquetas 1..6)
% X_test   3000x32x32x3 uint8, y_test   3000x1, class_names cell 1x6
```

## Tareas

### 1. Preprocesado

Aplana cada imagen de `32×32×3` a un vector de 3072 elementos y
normaliza a `[0, 1]`. En color: no conviertas a escala de grises.

### 2. Clasificador Bayesiano Gaussiano

Naive Bayes Gaussiano asume que, dada la clase, cada píxel es
independiente de los demás y sigue una normal con media y varianza
propias de esa clase y esa dimensión.

```matlab
mdl = fitcnb(X_train, y_train);   % distribucion normal por defecto
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

### 4. Correlación entre píxeles

Calcula la matriz de correlación de los píxeles de entrenamiento de una
sola clase con `corrcoef` (coeficiente de correlación, no covarianza
bruta, para poder compararlo con el espacio de PCA del apartado 5) y
calcula el valor absoluto medio de los elementos fuera de la diagonal.

### 5. PCA antes de Naive Bayes

Proyecta los datos a 50 componentes principales y repite el
entrenamiento y la evaluación.

```matlab
[coeff, scoreTrain, ~, ~, ~, mu] = pca(X_train, 'NumComponents', 50, 'Algorithm', 'eig');
scoreTest = (X_test - mu) * coeff;
```

Recoge en una tabla el accuracy y el tiempo con PCA y sin PCA.

Calcula además, con el mismo procedimiento del apartado 4, el valor
absoluto medio fuera de la diagonal de la correlación en el espacio de
PCA en dos casos: sobre todo el conjunto de entrenamiento, y sobre una
sola clase. Añade los dos números a la tabla.

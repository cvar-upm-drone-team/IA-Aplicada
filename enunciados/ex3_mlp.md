# Ejercicio 3 — Clasificación con un Perceptrón Multicapa (MLP)

Clasificar imágenes de 6 categorías con un **MLP** (red totalmente
conectada), medir el efecto de variar la arquitectura y el de aplicar
PCA.

## Dataset

Descarga `cifar_subset.mat` de la [release del repositorio](https://github.com/GPatiA2/MUIA-IA_Aplicada/releases/tag/dataset-release-v1)
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
Convierte las etiquetas a `categorical` para Deep Learning Toolbox.

### 2. Arquitectura y entrenamiento

```matlab
layers = [
    featureInputLayer(3072)
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(6)
    softmaxLayer
];
```

Entrena con `trainnet` y optimizador Adam, 50–80 épocas, reservando
parte del entrenamiento como validación
(`trainingOptions(..., 'ValidationData', ...)`).

Representa la curva de pérdida de entrenamiento y validación por época,
a partir de la segunda salida de `trainnet`
(`info.TrainingHistory` e `info.ValidationHistory`).

**Dos avisos de rendimiento:**

- Un `MiniBatchSize` de 32 genera unos 320 mini-lotes por época, cada
  uno con su coste fijo de lanzamiento en GPU. Usa 128–256.
- `ValidationFrequency` se cuenta en **iteraciones, no en épocas**. Con
  `MiniBatchSize` 256 hay ~40 iteraciones por época, así que un valor
  de 10 lanza ~4 validaciones por época. Para validar una vez por época,
  pásale `ceil(nTrain / miniBatchSize)`.

### 3. Evaluación

`utils/evaluar.m` calcula accuracy, matriz de confusión,
precision/recall/F1 (por clase y macro-promediadas), AUC y AP
One-vs-Rest. Definiciones en [`docs/metrics.md`](../docs/metrics.md).

```matlab
addpath('utils');
scores = minibatchpredict(net, X_test);
preds  = scores2label(scores, categories(Y_train));
evaluar(Y_test, preds, scores, class_names);
```

### 4. Variaciones de la arquitectura

Entrena al menos dos variantes más, cambiando el número de neuronas de
las capas ocultas, añadiendo o quitando una capa, o añadiendo un
`dropoutLayer`. Recoge en una tabla el accuracy de test de cada variante
junto al de la arquitectura base.

### 5. PCA antes del MLP

Proyecta los datos a 50 componentes principales y repite el
entrenamiento con la misma arquitectura, ajustando `featureInputLayer` a
50.

```matlab
[coeff, scoreTrain, ~, ~, ~, mu] = pca(X_train, 'NumComponents', 50, 'Algorithm', 'eig');
scoreTest = (X_test - mu) * coeff;
```

Recoge en la tabla el accuracy y el tiempo de entrenamiento con PCA y
sin PCA.

# Ejercicio 4 — Clasificación con una Red Convolucional (CNN)

Clasificar imágenes de 6 categorías con una **CNN**, que trabaja
directamente sobre la estructura 2D de la imagen en vez de aplanarla.

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

Aquí **no** se aplana la imagen: la CNN trabaja sobre la estructura 2D
con sus 3 canales de color.

- Convierte a `single` y normaliza a `[0, 1]`.
- Reordena al formato `32×32×3×N` que espera `imageInputLayer`: en el
  `.mat` vienen como `N×32×32×3`, así que `permute(X, [2 3 4 1])`
  (vectorizado, no un bucle por imagen).
- Convierte las etiquetas a `categorical`.

### 2. Arquitectura y entrenamiento

Tres bloques convolucionales que doblan los canales mientras el *pooling*
parte el mapa espacial por la mitad (`32×32 → 16×16 → 8×8 → 4×4`), y un
clasificador denso al final:

```matlab
layers = [
    imageInputLayer([32 32 3], 'Normalization', 'none')

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(6)
    softmaxLayer
];
```

Entrena con `trainnet` y Adam, con estos valores de partida:
`InitialLearnRate` 5e-4, `MiniBatchSize` 64, 150 épocas,
`'LearnRateSchedule', 'piecewise'`, un 15% del train como validación y
`'OutputNetwork', 'best-validation'`.

Aplica *data augmentation* sobre el conjunto de entrenamiento (flip
horizontal y traslación de unos pocos píxeles) con `imageDataAugmenter`
+ `augmentedImageDatastore`.

**Tres avisos de rendimiento:**

- `FillValue` de `imageDataAugmenter` vale 0 (negro) por defecto, así que
  la traslación mete bordes negros artificiales que empeoran el
  resultado. Usa `'FillValue', 0.5` (gris neutro), acorde a la
  normalización `[0, 1]`.
- `ValidationFrequency` se cuenta en **iteraciones, no en épocas**. Con
  `MiniBatchSize` 64 hay ~160 iteraciones por época, así que un valor de
  10 lanza ~16 validaciones por época en vez de 1, y puede multiplicar
  varias veces el tiempo total de entrenamiento sin que nada falle de
  forma visible. Para validar una vez por época, pásale
  `ceil(nTrain / miniBatchSize)`.
- Haz la augmentation con operaciones vectorizadas sobre todo el lote,
  nunca con un bucle imagen por imagen.

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

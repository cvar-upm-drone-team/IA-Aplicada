# MUIA/MUAR — IA Aplicada: ejercicios de clasificación de imágenes

Cuatro ejercicios independientes de clasificación de imágenes (KNN,
Bayesiano, MLP, CNN) para la asignatura de IA Aplicada. Se entregan en
MATLAB.

**→ Versión más cómoda de leer: [sitio del curso](https://cvar-upm-drone-team.github.io/IA-Aplicada/)**

## Estructura

```
enunciados/              Un enunciado .md por ejercicio (independientes)
  ex1_knn.md  ex2_bayes.md  ex3_mlp.md  ex4_cnn.md
docs/
  dataset.md             Dataset: descarga, formato, uso en Colab
  metrics.md             Definiciones de precision/recall/F1/AUC/AP
  faq.md                 Problemas previsibles de instalación de MATLAB
utils/
  evaluar.m              Evaluación común: accuracy, confusión, P/R/F1, AUC, AP
```

## Dataset

CIFAR-10, subset de 6 clases (avión, automóvil, pájaro, gato, ciervo,
perro), **2000 train + 500 test por clase** (pensado para entrenar en
minutos aunque no tengas tarjeta gráfica).

El fichero que necesitáis es **`cifar_subset.mat`**, y se descarga de
la release del repositorio:

**→ [Descargar cifar_subset.mat](https://github.com/cvar-upm-drone-team/IA-Aplicada/releases/tag/dataset-release-v1)**

Colocadlo en `data/cifar_subset.mat` dentro del repositorio, que es
donde lo buscan los enunciados. Todos los detalles —formato, uso en
Google Colab, verificación de la descarga— en
[`docs/dataset.md`](docs/dataset.md).

## Los ejercicios

| | Enunciado | Método | Idea central |
|---|---|---|---|
| 1 | [`ex1_knn.md`](enunciados/ex1_knn.md) | K-Nearest Neighbors | Clasificar por parecido directo entre vectores de píxeles |
| 2 | [`ex2_bayes.md`](enunciados/ex2_bayes.md) | Naive Bayes Gaussiano | Qué pasa cuando su suposición de independencia es falsa |
| 3 | [`ex3_mlp.md`](enunciados/ex3_mlp.md) | Perceptrón multicapa | Aprender una frontera no lineal sobre la imagen aplanada |
| 4 | [`ex4_cnn.md`](enunciados/ex4_cnn.md) | Red convolucional | Explotar la estructura espacial 2D en vez de aplanar la imagen |

Cada ejercicio se entrega por separado; ninguno necesita que hayas
hecho los anteriores. Los tres primeros comparten preprocesado
(aplanar la imagen a 3072 valores en color) y terminan con una llamada
a `utils/evaluar.m` para la evaluación; la CNN (ejercicio 4) trabaja
directamente sobre la imagen sin aplanarla.

### La función de evaluación

`utils/evaluar.m` se os da hecha para que todos midáis lo mismo de la
misma forma: accuracy, matriz de confusión, precision/recall/F1 (por
clase y macro-promediadas), AUC y AP One-vs-Rest. Acepta etiquetas
numéricas (`1..6`) o `categorical`:

```matlab
addpath('utils');
evaluar(y_test, preds, scores, class_names);
```

`scores` es la matriz de puntuaciones por clase — la segunda salida de
`predict`, o la salida de `minibatchpredict` en los ejercicios con
redes. Las definiciones de cada métrica están en
[`docs/metrics.md`](docs/metrics.md).

### Antes de empezar

- Descarga el dataset (ver arriba) y colócalo en `data/cifar_subset.mat`.
- **Usa las imágenes en color**, no las conviertas a gris: baja el
  accuracy en los cuatro métodos sin ahorrar tiempo de entrenamiento
  salvo en KNN y Bayes, que ya tardan segundos.
- Separa una parte del train como **validación** si vas a ajustar
  hiperparámetros. El test es solo para el número final.
- En MATLAB, cuidado con `ValidationFrequency`: se cuenta en
  iteraciones, no en épocas (aviso detallado en el enunciado del
  ejercicio 4). Es un error que no da ningún síntoma salvo que el
  entrenamiento tarde mucho más de lo normal.
- Si MATLAB se queja de una función que no existe (`fitcknn`,
  `trainnet`...), probablemente te falte instalar una toolbox, no un
  error de tu código: mira [`docs/faq.md`](docs/faq.md).

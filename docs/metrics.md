# Métricas de evaluación

La **accuracy** (proporción de aciertos global) es fácil de calcular pero
puede **esconder problemas graves** cuando el clasificador se comporta muy
distinto en unas clases que en otras. Por eso, además de accuracy y la
matriz de confusión, en cada ejercicio se pide calcular estas métricas
adicionales sobre el conjunto de test.

## Precision, Recall y F1 (por clase)

Para una clase `c`, a partir de la matriz de confusión:

- **Precision** = TP / (TP + FP) — fracción de acierto entre las veces
  que el modelo predijo la clase `c`.
- **Recall** = TP / (TP + FN) — fracción de las imágenes que realmente
  son de la clase `c` que el modelo llegó a detectar.
- **F1** = media armónica de precision y recall:
  `F1 = 2 · precision · recall / (precision + recall)`

donde TP (verdaderos positivos), FP (falsos positivos) y FN (falsos
negativos) se calculan para esa clase tratándola como "positiva" frente
a "el resto".

### Macro-average

Para tener un único número que resuma las 6 clases, se calcula la
**media simple** (no ponderada) de precision/recall/F1 de las 6 clases.
A diferencia de la accuracy global, el macro-average **no se deja
dominar por las clases en las que el modelo va mejor**: si el modelo
ignora completamente una clase, eso penaliza igual que si ignorara
cualquier otra, independientemente de cuántos ejemplos tenga esa clase
en el test (aquí las 6 clases tienen el mismo número de ejemplos, 150,
así que no cambia el resultado respecto a una media ponderada — pero sí
sería distinto en un dataset desbalanceado).

### Por qué importa

Es perfectamente posible que un modelo tenga una accuracy "decente" y, a
la vez, **ignore casi por completo una clase concreta**: puede tener
precision alta pero recall muy bajo en esa clase (casi nunca la predice,
pero cuando lo hace acierta), o al revés. La accuracy global no deja ver
esto — hace falta mirar precision/recall por clase.

## Curva ROC y AUC (One-vs-Rest)

Para un problema multiclase, la curva ROC y el área bajo ella (AUC) se
calculan **por clase**, tratando esa clase como "positiva" y el resto
como "negativa" (enfoque **One-vs-Rest**, OvR), usando las probabilidades
que da el modelo para esa clase (no solo la predicción final). El AUC
mide qué tan bien el modelo **ordena** los ejemplos de esa clase por
encima de los del resto, independientemente del umbral de decisión.
Luego se promedia (macro) el AUC de las 6 clases.

- AUC = 0.5 → el modelo no discrimina mejor que el azar para esa clase.
- AUC = 1.0 → separación perfecta.

## "mAP" (Average Precision, One-vs-Rest)

**Importante**: mAP (*mean Average Precision*) es una métrica propia de
tareas de **detección de objetos o recuperación de información**, donde
hay que ordenar/rankear muchos candidatos por imagen o por consulta. En
un problema de **clasificación de una sola etiqueta por imagen** como
este, mAP en su sentido estricto no aplica directamente.

Lo que sí se puede calcular, y que tiene un "sabor" parecido, es la
**Average Precision (AP) por clase** a partir de la curva
precision-recall de esa clase frente al resto (de nuevo, enfoque OvR,
usando las probabilidades del modelo), promediada entre las 6 clases.
A esto nos referiremos como **"AP macro" o "mAP-like"** en los
ejercicios — es una buena práctica para familiarizarse con el concepto,
pero hay que tener claro que no es el mAP que se usaría, por ejemplo,
para evaluar un detector de objetos.

**Nota técnica (MATLAB vs. Python)**: en Python, `average_precision_score`
calcula el AP con la fórmula de interpolación por escalones estándar en
detección/recuperación. En MATLAB, `perfcurve` con `XCrit='tpr'` y
`YCrit='ppv'` da el área bajo la curva precision-recall por la **regla
trapezoidal**, que es un cálculo distinto. Ambos son razonables, pero
**no dan exactamente el mismo número** — la diferencia es pequeña cuando
las probabilidades del modelo están bien calibradas y la curva
precision-recall es suave (p. ej. en la CNN), pero puede ser bastante
mayor cuando el modelo da probabilidades peor calibradas y la curva es
más irregular (p. ej. en Naive Bayes). No es un error: es una diferencia
real de metodología entre ambas implementaciones. El AUC (ROC) y las
métricas de precision/recall/F1, en cambio, sí coinciden entre ambos
lenguajes porque se calculan de la misma forma.

## Qué se pide en cada ejercicio

Para el clasificador que te toque (KNN / Bayesiano / MLP / CNN):

1. Calcula **precision, recall y F1 por clase**, y su macro-average.
2. Calcula el **AUC (One-vs-Rest, macro)** a partir de las probabilidades
   que da tu modelo para cada clase.
3. Calcula el **"AP macro" (One-vs-Rest)** de la misma forma.
4. Identifica si hay alguna clase con **precision y recall muy
   distintos** entre sí (por ejemplo, precision alta pero recall bajo).
   Explica qué significa eso sobre el comportamiento del modelo en esa
   clase, y por qué la accuracy global no lo habría dejado ver.

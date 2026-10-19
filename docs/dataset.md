# Dataset: CIFAR-10 (subconjunto reducido)

Este dataset es un subconjunto de [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
preparado para los ejercicios de clasificación de imágenes de IA Aplicada
(MUIA/MUAR).

## Clases

6 clases, codificadas como enteros **1 a 6** (convención MATLAB):

| Etiqueta | Clase |
|---|---|
| 1 | avión (`airplane`) |
| 2 | automóvil (`automobile`) |
| 3 | pájaro (`bird`) |
| 4 | gato (`cat`) |
| 5 | ciervo (`deer`) |
| 6 | perro (`dog`) |

## Tamaño

- **Entrenamiento**: 5000 imágenes/clase → 30000 imágenes (el máximo disponible
  en CIFAR-10 para estas 6 clases)
- **Test**: 1000 imágenes/clase → 6000 imágenes (también el máximo disponible)
- Imágenes de **32×32 píxeles, color (RGB)**

## Variante reducida (para quien no tenga GPU)

Existe una segunda versión, **`cifar_subset_15k.mat`**, con 2000
train + 500 test por clase (12000 + 3000 = 15000 imágenes, un 42% del
tamaño del dataset completo). Mismo formato, mismas variables, mismo
convenio de etiquetas — solo cambia la cantidad de imágenes.

Está pensada para quien no tenga tarjeta gráfica: entrenar la CNN del
ejercicio 4 en CPU con el dataset completo puede rondar la media hora;
con esta variante baja a unos minutos, a cambio de entre 1 y 3 puntos
menos de accuracy (medido con las soluciones de referencia). El orden
de dificultad entre métodos se mantiene igual con cualquiera de las dos.

**Importante**: los enunciados cargan siempre `data/cifar_subset.mat`
por nombre fijo. Para usar la variante reducida, descárgala y
**renómbrala a `cifar_subset.mat`** al colocarla en `data/` — no se
puede tener las dos a la vez con sus nombres originales sin editar el
código.

## Cómo obtener el dataset

El fichero que necesitas es **`cifar_subset.mat`** (106 MB).

**→ Descárgalo de la release:** https://github.com/cvar-upm-drone-team/IA-Aplicada/releases/tag/dataset-release-v1

No está dentro del árbol de git porque pesa 105,7 MiB, por encima del
límite de 100 MiB que GitHub impone a los ficheros versionados; de ahí
que se distribuya como release. La variante reducida
(`cifar_subset_15k.mat`) está en la misma release como segundo *asset*.

Una vez descargado, colócalo en **`data/cifar_subset.mat`** dentro del
repositorio: es la ruta donde lo buscan los enunciados
(`load('data/cifar_subset.mat')`). Si vas a trabajar en Google Colab,
súbelo a tu Google Drive y móntalo desde el notebook (ver más abajo).

Verifica la integridad tras descargarlo:

```bash
sha256sum cifar_subset.mat
# 3cb1a3e2969955bd8f30832e96e047d6b4141014bcd0fe12ef8ac80b67369a7c

sha256sum cifar_subset_15k.mat
# 5a3fd2cd492d19efbe8b208139051e64735fa205f1d66da5e981dad6275c0101
```

En Windows, desde PowerShell:

```powershell
Get-FileHash cifar_subset.mat -Algorithm SHA256
```

Desde MATLAB, comprobando el tamaño (no uses `mlreportgen.utils.hash`:
no devuelve el hash del contenido del fichero y no coincide con ninguna
herramienta estándar):

```matlab
d = dir('cifar_subset.mat');
d.bytes    % debe ser 110880824
```

### Verificar que ha salido bien

```matlab
load('data/cifar_subset.mat');
size(X_train)      % 30000  32  32   3
size(y_train)      % 30000   1
unique(y_train)'   % 1 2 3 4 5 6
histcounts(y_train, 1:7)   % 5000 cada clase
class_names        % 6 nombres, en orden de etiqueta
```

## Uso en Google Colab

El fichero pesa 106 MB y el almacenamiento de Colab se borra al cerrar la
sesión, así que **conviene no volver a descargarlo cada vez**. La vía
recomendada es subirlo una vez a Google Drive y montar Drive desde el
notebook:

1. **Una sola vez**: descarga `cifar_subset.mat` de la release (enlace
   arriba) y súbelo a tu Google Drive, por ejemplo a
   `MiUnidad/IA_Aplicada/cifar_subset.mat`.
2. **En cada notebook**, monta Drive al principio:

```python
from google.colab import drive
drive.mount('/content/drive')

MAT_PATH = '/content/drive/MyDrive/IA_Aplicada/cifar_subset.mat'
```

3. Carga el `.mat` desde esa ruta:

```python
from scipy.io import loadmat
d = loadmat(MAT_PATH)
X_train, y_train = d['X_train'], d['y_train'].ravel()   # etiquetas 1..6
X_test,  y_test  = d['X_test'],  d['y_test'].ravel()
class_names = [c[0] for c in d['class_names'][0]]
```

Así la descarga del dataset ocurre una vez y no cuenta en el tiempo de
cada práctica.

## Fichero

`data/cifar_subset.mat`, con las siguientes variables:

| Variable | Tipo / tamaño | Descripción |
|---|---|---|
| `X_train` | `uint8`, `30000×32×32×3` | Imágenes de entrenamiento |
| `y_train` | `double`, `30000×1` | Etiquetas de entrenamiento (1–6) |
| `X_test` | `uint8`, `6000×32×32×3` | Imágenes de test |
| `y_test` | `double`, `6000×1` | Etiquetas de test (1–6) |
| `class_names` | `cell` de 6 strings | Nombres de las clases, en el mismo orden que las etiquetas |

### Cargar el dataset en MATLAB

```matlab
load('data/cifar_subset.mat');

% Ver una imagen de ejemplo
imshow(squeeze(X_train(1,:,:,:)));
title(class_names{y_train(1)});
```

### Notas importantes

- Las imágenes vienen en **`uint8` (valores 0–255)**. La mayoría de métodos
  (KNN, Bayesiano, MLP) esperan entradas en `double` y normalizadas
  (por ejemplo, dividiendo por 255). Las funciones de Deep Learning Toolbox
  (`trainNetwork`/`trainnet`) aceptan `uint8` pero conviene normalizar igual.
- Para KNN/Bayesiano/MLP "clásico" se recomienda trabajar con las imágenes
  **aplanadas pero en color** (vector de 32×32×3 = 3072 valores): estos
  métodos tratan la imagen como un vector de características, no
  explotan la estructura espacial 2D, pero el color sí es información
  relevante para distinguir estas clases y no hay motivo para
  descartarlo (convertir a gris solo tira información útil). La CNN, en
  cambio, sí explota tanto el color como la estructura 2D, trabajando
  directamente sobre la imagen `32×32×3` sin aplanar.
- El dataset ya viene separado en train/test — no hace falta hacer el split
  vosotros, pero sí podéis (y se recomienda) separar una parte del train
  como validación si vais a ajustar hiperparámetros.

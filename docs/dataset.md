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

- **Entrenamiento**: 2000 imágenes/clase → 12000 imágenes
- **Test**: 500 imágenes/clase → 3000 imágenes
- Imágenes de **32×32 píxeles, color (RGB)**

Es un subconjunto reducido pensado para que los cuatro ejercicios,
incluida la CNN, entrenen en minutos aunque no tengas tarjeta gráfica.

## Cómo obtener el dataset

El fichero que necesitas es **`cifar_subset.mat`** (44 MB).

**→ Descárgalo de la release:** https://github.com/cvar-upm-drone-team/IA-Aplicada/releases/tag/dataset_release

No está dentro del árbol de git porque los ficheros binarios de este
tamaño no se versionan bien en git; de ahí que se distribuya como
release.

Una vez descargado, colócalo en **`data/cifar_subset.mat`** dentro del
repositorio: es la ruta donde lo buscan los enunciados
(`load('data/cifar_subset.mat')`). Si vas a trabajar en Google Colab,
súbelo a tu Google Drive y móntalo desde el notebook (ver más abajo).

Verifica la integridad tras descargarlo:

```bash
sha256sum cifar_subset.mat
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
d.bytes    % debe ser 46200824
```

### Verificar que ha salido bien

```matlab
load('data/cifar_subset.mat');
size(X_train)      % 12000  32  32   3
size(y_train)      % 12000   1
unique(y_train)'   % 1 2 3 4 5 6
histcounts(y_train, 1:7)   % 2000 cada clase
class_names        % 6 nombres, en orden de etiqueta
```

## Uso en Google Colab

El fichero pesa 44 MB y el almacenamiento de Colab se borra al cerrar la
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
| `X_train` | `uint8`, `12000×32×32×3` | Imágenes de entrenamiento |
| `y_train` | `double`, `12000×1` | Etiquetas de entrenamiento (1–6) |
| `X_test` | `uint8`, `3000×32×32×3` | Imágenes de test |
| `y_test` | `double`, `3000×1` | Etiquetas de test (1–6) |
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

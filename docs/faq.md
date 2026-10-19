# FAQ: instalación y entorno de MATLAB

## Qué hace falta instalar

MATLAB se vende por módulos: el programa base trae lo básico y cada
área (estadística, redes neuronales, imagen...) va en un módulo aparte
que se llama *toolbox*. Para estos ejercicios hacen falta dos:

| Toolbox | Hace falta para | Ejercicios |
|---|---|---|
| **Statistics and Machine Learning** | KNN, Naive Bayes, PCA y el cálculo de las métricas | **los cuatro** |
| **Deep Learning** | entrenar redes neuronales | 3 y 4 |

Ojo con la primera fila: aunque los ejercicios 3 y 4 son de redes
neuronales, **también necesitan la toolbox de estadística**, porque el
script de evaluación (`utils/evaluar.m`) la usa para calcular las
métricas. Es fácil asumir que con la de Deep Learning basta para esos
dos, y no es así.

## Cómo comprobar si la tienes

Escríbelo en la ventana de comandos de MATLAB:

```matlab
exist('fitcknn')     % 2 = la toolbox de estadística está puesta, 0 = no
exist('trainnet')    % 2 = la de Deep Learning está puesta, 0 = no
```

Si quieres la lista completa de lo que tienes instalado, el comando es
`ver`.

**Cuidado con una trampa**: existe otro comando, `license('test', ...)`,
que parece servir para lo mismo pero no. Ese dice si *la licencia*
permite usar la toolbox, no si está instalada. Puede decir que sí
mientras la toolbox no está puesta, y entonces nada funciona. Fíate de
`exist`.

## El error que verás si te falta una toolbox

```
Undefined function 'fitcknn' for input arguments of type 'double'.
```

Suena a error de programación, pero no lo es: significa "esa función no
existe en este MATLAB", o sea, te falta la toolbox. **No hace falta
reinstalar MATLAB**: se añade desde el propio MATLAB, en el menú de
arriba, **Home → Add-Ons → Get Add-Ons**, buscando la toolbox por su
nombre e instalándola.

La licencia de estudiante normalmente las cubre, pero en el instalador
hay que marcarlas a mano: **por defecto no vienen todas seleccionadas**,
y de ahí viene la mayoría de estos casos.

## Versión de MATLAB

Este material usa `trainnet` para entrenar redes, que existe **desde
R2023b**. Si tienes una versión anterior verás:

```
Undefined function 'trainnet'
```

y no es que te falte una toolbox: es que tu MATLAB es demasiado
antiguo. En esas versiones el comando se llamaba `trainNetwork` y la
red se construía de forma algo distinta, así que tendrías que adaptar
el código del enunciado. Antes de buscar el fallo en otra parte,
comprueba siempre qué versión tienes, porque este caso se confunde con
mucha facilidad con una toolbox que falta: el mensaje de error es
prácticamente el mismo.

## ¿Hace falta tarjeta gráfica?

No. Todo funciona sin GPU; lo único que cambia es el tiempo. El dataset
de este curso ya está pensado para entrenar en minutos incluso sin
tarjeta gráfica.

Por defecto, `trainnet` usa la tarjeta gráfica automáticamente si la
detecta (la opción `ExecutionEnvironment` de `trainingOptions` vale
`'auto'` de serie), así que normalmente no tienes que hacer nada. Si
quieres comprobar o forzar que la use:

```matlab
gpuDeviceCount   % cuantas tarjetas ve MATLAB (0 = ninguna)
gpuDevice        % datos de la que usaría por defecto
```

Si `gpuDeviceCount` da 0 aunque tengas una tarjeta NVIDIA, lo más
probable es que falte **Parallel Computing Toolbox** (comprúebalo con
`exist('gpuDevice')`, igual que con las demás toolboxes) o que los
drivers de la tarjeta estén desactualizados.

Para forzar explícitamente el uso de la GPU (y que falle con un error
claro si no la encuentra, en vez de seguir en CPU en silencio):

```matlab
options = trainingOptions('adam', ..., 'ExecutionEnvironment', 'gpu');
```

Y para forzar CPU, por ejemplo si quieres comparar tiempos:

```matlab
options = trainingOptions('adam', ..., 'ExecutionEnvironment', 'cpu');
```

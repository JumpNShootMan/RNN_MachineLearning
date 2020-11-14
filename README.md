# Tarea Académica 2 - Estudio de las Redes Neuronales Reurrentes
## Curso - Machine Learning
### Elaborado por:
- Daniel Núñez
- Diego Urrutia

### Tarea Académica que estudia las RNNs y sus usos. Se elaborará durante las siguientes semanas.

## 1. Introcucción y Motivación

## 2. Intuición detrás de la idea (cómo funciona)

## 3. Qué bibliotecas implementan la red y cuál es la sugerida (sklearn, tensorflow, pytorch, caffe, etc?)

Para la elaboración de una solución que implemente redes neuronales recurrentes, Keras de TensforFlow es la librería sugerida. Esta librería facilita la creación de redes neuronales en general, y cuenta con capas de redes neuronales recurrentes listas para aplicar en la librería keras.layers (.RNN/.LSTM/.GRU). Como afirman Chung et al. (2014), las LSTM (Long Short Term Memory) y GRU (Gated Recurrent Units) son capas de redes neuronales recurrentes más especializadas para mantener un aprendizaje de las dependencias a largo plazo al mantener la información en su memoria por tiempos más prolongados. Por un lado, las LSTM cuentan con una unidad de memoria capaz de controlar el flujo de información, mientras que las GRU realizan la misma tarea al reducir la complejidad de la arquitectura LSTM (compuertas de 'entrada', 'salida' y 'olvidar'), simplemente ofreciendo el contenido de la memoria en su totalidad.

En base a este conocimiento, al implementar una solución se recomienda utilizar la capa LSTM de la librería keras.layers.LSTM para conseguir los mejores resultados, pues su complejidad permite captar las relaciones a largo plazo de una manera más compleja y precisa. Aún así, también se recomienda utilizar la capa GRU debido a su eficiencia, pues estas cuentan con un rendimiento similar a las LSTM y no requieren tanto tiempo de entrenamiento por su arquitectura más simplificada.

En síntesis, para la elaboración de una solución de redes neuronales recurrentes básica, se necesitan importar las siguientes librerías. Esto incluye un conjunto de datos.

``` python

import tensorflow as tf
from tensorflow import keras #de aquí también se puede obtener el dataset keras.dataset.mnist
from tensorflow.keras import layers # layers.RNN, layers.LSTM, layers.GRU
import numpy as np #procesamiento de información con numpy
import matplotlib.pyplot as plt #visualización de la información procesada
```

## 4. 3 ejemplos descriptivos de aplicaciones de la misma, pueden incluir videos de terceros que muestren aplicaciones interesantes.

### - Modelos de análisis de textos

### - Modelos de reconocimiento de voz

### - Modelos generativos

## 5. Conclusiones del trabajo

## 6. Referencias (links a los papers y artículos introductorios)

Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. http://arxiv.org/abs/1412.3555



# Tarea Académica 2 - Estudio de las Redes Neuronales Recurrentes
## Curso - Machine Learning
### Elaborado por:
- Daniel Núñez
- Diego Urrutia

### Tarea Académica que estudia las RNNs y sus usos. Se elaborará durante las siguientes semanas.

## 1. Introducción y Motivación

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

## 4. Tres ejemplos descriptivos de aplicaciones de la misma, pueden incluir videos de terceros que muestren aplicaciones interesantes.

### Modelos de análisis de textos más ejemplo por Luo & Chen (2020)

### Modelos de reconocimiento de voz más ejemplo por Kumar & Aggarwal (2020)

### Modelos generativos más ejemplo por Hernandez et al. (2020)

## 5. Conclusiones del trabajo

Las redes neuronales recurrentes son sumamente importantes para el análisis de patrones dependientes del tiempo. Esta técnica permite el análisis de una forma más humana, en donde se toma en consideración el orden de los elementos para determinar una representación más precisa de lo que se está leyendo. Al aproximarnos al final de la carrera, han surgido diversas ideas con respecto a soluciones que podríamos implementar en el país para solucionar distintos tipos de problemáticas. Como estudiantes de Ciencias de la Computación, un mejor conocimiento de los métodos de aprendizajes de máquinas nos va a ayudar a comprender qué propuestas o soluciones deberíamos utilizar para un respectivo problema y, además, nos contextualiza el proceso evolutivo que han tenido las redes neuronales a lo largo del tiempo. Gracias a esta investigación comprendemos la importancia que tienen las redes neuronales recurrentes en problemas de predicción de secuencias a nivel de reconocimiento de texto y voz junto con su aplicación en modelos generativos de datos. Sumado a ello, también comprendemos la importancia de los avances en las redes neuronales recurrentes para prevenir el problema de desvanecimiento de gradiente y reducir los tiempos de entrenamiento con los modelos LSTM y GRU que consideran elementos de memoria (unidad de memoria y compuertas de información) y diferencias en arquitectura (las GRU reducen la complejidad de esta para reducir los tiempos de entrenamiento) para aumentar la exactitud y eficiencia de las predicciones.


## 6. Referencias (links a los papers y artículos introductorios)

- Papers

Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. http://arxiv.org/abs/1412.3555

Hernandez-Matamoros, A., Fujita, H., & Perez-Meana, H. (2020). A novel approach to create synthetic biomedical signals using BiRNN. Information Sciences, 541, 218–241. https://doi.org/10.1016/j.ins.2020.06.019

Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. http://arxiv.org/abs/1412.3555

Kumar, A., & Aggarwal, R. K. (2020). Discriminatively trained continuous Hindi speech recognition using integrated acoustic features and recurrent neural network language modeling. Journal of Intelligent Systems, 30(1), 165–179. https://doi.org/10.1515/jisys-2018-0417

- Artículos Web introductorios y repositorios de información para las librerías

Recurrent Neural Networks (RNN) with Keras  |  TensorFlow Core. (n.d.). Retrieved November 13, 2020, from https://www.tensorflow.org/guide/keras/rnn

Donges, N. (2019). A Guide to RNN: Understanding Recurrent Neural Networks and LSTM. https://builtin.com/data-science/recurrent-neural-networks-and-lstm

Brownlee, J. (2018). When to Use MLP, CNN, and RNN Neural Networks. https://machinelearningmastery.com/when-to-use-mlp-cnn-and-rnn-neural-networks/

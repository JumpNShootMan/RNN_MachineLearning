# Tarea Académica 2 - Estudio de las Redes Neuronales Recurrentes

## El archivo a ser evaluado es el colab adjunto en el repositorio, tal como se indica en el enunciado. La finalidad del readme fue para organizarnos e ir formando una plantilla de avances mientras acababamos el colab.


## Curso - Machine Learning
### Elaborado por:
- Daniel Núñez
- Diego Urrutia

### Tarea Académica que estudia las RNNs y sus usos. Se elaborará durante las siguientes semanas.

## 1. Introducción y Motivación

### **Introducción**

Las redes neuronales recurrentes (RNN), son un tipo de red neuronal que permite hacer procesamientos de datos de naturaleza secuencial, es decir, datos en los cuales el orden es fundamental. Lo especial de estas redes neuronales es que son diferentes a la arquitectura de la red neuronal tradicional. Estas están conectadas entre si y no solo reciben un único output. 


En la siguiente imagen, se observa que la mayoría de las redes son feed-forward, pasan por una ruta directa, mientras que en las RNN hay bucles que permiten “recordar” los datos de en la red.

![](https://miro.medium.com/proxy/0*mRHhGAbsKaJPbT21.png)

### **Motivación**

#### Evolución de las Redes Neuronales en Ciencias de la Computación
*    1958 – Perceptron
*    1965 – Multilayer Perceptron
*    1980’s
        * Neuronas Sigmoidales
        * Redes Feedforward
        * Backpropagation

*    1989 – Convolutional neural networks (CNN) / Recurent neural networks (RNN)
*    1997 – Long short term memory (LSTM)
*    2006 – Deep Belief Networks (DBN): Nace deep learning

        *  Restricted Boltzmann Machine
        *  Encoder / Decoder = Auto-encoder
*    2014 – Generative Adversarial Networks (GAN)

Las redes neuronales existen desde 1958, pero en estos últimos años están tomando gran relevancia, ya que se tiene los recursos computacionales para poder aplicarlas, cosa que en 1958 no se podían realizar estudios mas profundos de estas redes. 

Las redes neuronales tradicionales vienen de la idea de imitar las funciones de las neuronas de los organismos vivos, a partir de detectar características.Las redes neuronales recurrentes,también imitan el funcionamiento de las neuronas, pero ya se introduce lo que es la "memoria". Puede decirse que las redes tradicionales son combinación de características, mientras que las RNN son combinaciones de múltiples características donde la secuencia de los datos y su correlación son fundamentales. 

Por ejemplo, si una red neuronal convolucional se le presenta una imagen o palabra, esta podrá clasificarla de manera correcta, pero que sucede si en vez de una imagen o palabra, la entrada es un video o una conversación. No sería posible, ya que estas redes solo funcionan un dato de entrada y de salida del mismo tamaño. En un video o conversación puede cambiar,además los datos estan correlacionados y se analiza el contexto.

## 2. Intuición detrás de la idea (cómo funciona)

### Ejemplo de una RNN:
![](https://miro.medium.com/max/2544/1*aIT6tmnk3qHpStkOX3gGcQ.png)

 Para MLP la salida es la siguiente fórmula:

![Alt text](Imagenes/f1.png?raw=true "Formula 1")

Sin embargo, en una RNN además de las entradas que ya conocemos, x, introduciremos las salidas de la capa oculta en el instante anterior, h, como entradas adicionales, las cuales irán multiplicadas por su propia matriz de pesos

![Alt text](Imagenes/f2.png?raw=true "Formula 2")

### Tipos de RNNs

Esta nueva arquitectura podemos llevar a cabo nuevas tareas dependiendo de la configuración de entradas/salidas que utilicemos

![](https://i.stack.imgur.com/b4sus.jpg)

La tipología *one-to-one* corresponde al MLP, el resto de tipologías nos permiten tareas como clasificación de secuencias o regresión a un valor (*many-to-one*) o regresión a varios valores futuros (*many-to-many*). 

* One to many: Tienes un input y varios outputs: Una imagen y la computadora va a describirla.

* Many to one : Varios inputs y un output : Datos numéricos de la rentabilidad de 2007-2012 , mi output es la rentabilidad para 2013.

* Many to many : Varios inputs y varios outputs: Un traductor para determinar el género. 

### Entrenamiento de las RNN : Backpropagation through time 

Las redes neuronales básicamente funcionan con Forward-Propagation para obtener la salida de tu modelo y comparar esta salida si es correcta o incorrecta para ajustar el error. Backpropagation es ir hacia atrás para encontrar las derivadas parciales del error respecto a los pesos, lo que le permite ir ajustando el modelo.Se busca intentar ajustar los pesos de tu modelo mientras entrena.


Backpropagation through time es un término para definir backpropagation en una RNN, ya que cumple la misma función, ajustar pesos.

En la imagen se observa el funcionamiento del backpropagation y el cálculo de sus derivadas parciales, cuando se desenrolla la red neuronal recurrente.


![](https://ars.els-cdn.com/content/image/1-s2.0-S0959438818302009-gr1.jpg)


Dentro de BPTT ,el error es propagado para cada instante de tiempo. Esto permite calcular el error y actualizar los pesos.

### Limitación  de las redes neuronales recuerrentes estándar :

Las RNN tienen limitaciones que afectan su eficacia :

Es necesario antes saber que es una gradiente

Un gradiente es una derivada parcial con respecto a sus entradas. 

En síntesis  , el gradiente indica el cambio en los pesos con respecto al cambio de en error. 

Las gradientes explosivas o exploding gradients ,suceden cuando el algoritmo asigna un alto numero en los pesos, sin razón y esto genera un problema en el entrenamiento.

Los gradientes desaparecidos o vanishing gradients , es otro problema y sucede cuando las gradientes son muy pequeños y el modelo deja de aprender. Estas ocurren mayormente cuando las series de tiempo son muy largas.


En la siguiente figura, se observa que mientras la serie es mas larga hay una pérdida de información.
![](https://cdn-images-1.medium.com/freeze/max/1000/1*lTeIFg5Ecl0hMd3FeNGDYA.png?q=20)


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
No se está utilizando sklearn o pytorch debido a que, por el lado de sklearn, la página no cuenta con documentación específica sobre redes neuronales recurrentes mientras que, por el otro lado, pytorch es una librería menos familiar para nosotros, y la mayoría de ejemplos o tutoriales que se pueden encontrar en línea están elaborados con keras de tensorflow.

## 4. Tres ejemplos descriptivos de aplicaciones de la misma, pueden incluir videos de terceros que muestren aplicaciones interesantes.

### Modelos de clasificación de textos más ejemplo por Luo & Chen (2020)

La primera aplicación de redes neuronales que vamos a describir es la de clasificación de textos. Este es un problema clásico en el dominio de procesamiento de lenguaje natural. Existen muchos usos para la clasificación de textos, como por ejemplo el análisis de sentimientos, etiquetas y más. En el caso de este ejemplo, se están utilizando las redes neuronales recurrentes (RNN) para analizar la calidad de textos con la finalidad de encontrar relaciones semánticas en la búsqueda de información.

Para lograr esto, el autor ha propuesto una innovación en el modelo clásico de RNN al agregar un mecanismo de atención, que se encarga de enfocar el análisis en secciones específicas de los datos de entrada al llevar a cabo predicciones. Además, se ha agregado una capa convolucional para reducir la dimensionalidad de los datos para reducir el tiempo de computación.

Para esta solución, se comienza ingresando los datos de entrada en orden. Seguido a esto, se llevan a cabo pasos en el tiempo que representan un ciclo en la RNN. Durante este proceso, se describen todos los cambios en un punto de tiempo respectivo, y esta información se almacena en un estado interno que participa junto con el estado de sistema. Para el análisis de estados internos, se lleva a cabo un proceso de recursión. En una perspectiva de estructura de árboles, el estado anterior es el padre del estado interno actual. Para calcular el estado interno actual, se analizan los estados anteriores y actuales, que son analizados por la RNN con mecanismo de atención para captar las relaciones semánticas entre las palabras para calcular conexiones basadas en contexto.

Seguido a esto, un nodo de salida aplica un clasificador de secuencia en donde se insertan los vectores de palabras y etiquetas de clasificación. El autor propone un cálculo basado en el promedio de todos los estados del sistema en base a un cálculo recursivo para generar los datos de salida. Finalmente, una vez que se tienen estas salidas, se analiza la calidad del texto en base a la capa convolucional, en donde se aplica un proceso de convolución y pooling para reducir la dimensionalidad de la data, se extraen las características y se obtienen valores óptimos. Para finalizar, una capa softmax lleva a cabo el proceso de predicción.

### Modelos de reconocimiento de voz más ejemplo por Kumar & Aggarwal (2020)


### Modelos generativos más ejemplo por Hernandez et al. (2020)

Esta aplicación de RNN es muy interesante, ya que se concentra en generar nueva información a partir de los patrones que ha identificado la red. La generación de nueva información a partir de reglas definidas manualmente nunca fue considerado una buena práctica, ya que no siempre se pueden asignar reglas, incluso aunque existan usuarios expertos en el tema. Es por esto que la generación de nueva información en base a patrones captados dinámicamente por un modelo de RNN es tan revolucionaria; puede asemejarse como nunca antes a datos reales.

Para el modelo generativo de los autores, la innovación proviene de la implementación de un modelo generativo basado en RNN bidireccionales para la generación de distintas señales biomédicas, como por ejemplo las de electrocardiogramas o eleectroencefalogramas. Estos datos pueden generarse para pacientes o eventos específicos, y puede repetirse el proceso para generar incluso más datos sintéticos.

Para iniciar, se lleva a cabo una etapa de preprocesamiento de datos, en donde se puede reducir el ruido de señales en base a una técnica de transformación de ondas discretas. Luego, se utiliza Z-Score para eliminar compensación y luego se segmenta y clusteriza la información de acuerdo a las anotaciones de la base de datos utilizada. En la etapa final del preprocesamiento se ajustan los tamaños de las señales para que todos tengan el mismo y estas se exportan para la RNN bidireccional (BiRNN).

La BiRNN combina dos RNN, en donde una analiza el dato desde el inicio y se mueve adelante en el tiempo y la otra comienza desde el final y analiza retrocediendo en el tiempo.

![Alt text](Imagenes/BiRNN.png?raw=true "Arquitectura BiRNN")

Las BiRNN van a aprender de secuencias que definen en una lista que describe el dato de entrada y la salida en un tiempo definido. Para el ejercicio se dividen las señales en dos grupos, en donde a uno de estos grupos se le agrega ruido a las señales. De esta manera, las dos señales atraviesan la capa de BiRNN para que se identifiquen secuencias que generen una salida de señales sintéticas: información nueva a partir de patrones encontrados para casos individuales.


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

Kostadinov, S. (2017). How Recurrent Neural Networks work | by Simeon Kostadinov | Towards Data Science. https://towardsdatascience.com/learn-how-recurrent-neural-networks-work-84e975feaaf7

Torres, J. (2019). Redes Neuronales Recurrentes - Jordi TORRES.AI. https://torres.ai/redes-neuronales-recurrentes/

Vanishing Gradients in Recurrent Neural Networks – mc.ai. (2018). https://mc.ai/vanishing-gradients-in-recurrent-neural-networks-2/

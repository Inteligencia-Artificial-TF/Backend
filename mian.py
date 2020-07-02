import numpy as np
import pandas as pd
import math
import copy
import time
from matplotlib import pyplot as plt
        

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)



errores = np.array(int)


class Backprogation:
    epocas = 10000
    const_aprendizaje = 0.1
    neuronasCapaEntrada = 25
    neuronasCapaOculta = 15
    neuronasCapaSalida = 1
    prediccion = 0
    salida = []
    def __init__(self, entradas, salidasEsperadas,prueba):
        self.entradas = entradas
        self.salidasEsperada = salidasEsperadas
        self.prueba = prueba
        self.pesos_capaOculta = np.random.uniform(size=(self.neuronasCapaEntrada,self.neuronasCapaOculta))
        self.bias_capaOculta =np.random.uniform(size=(1,self.neuronasCapaOculta))
        self.pesos_capaSalida = np.random.uniform(size=(self.neuronasCapaOculta,self.neuronasCapaSalida))
        self.bias_capaSalida = np.random.uniform(size=(1,self.neuronasCapaSalida))
    
    def mostar_datos_iniciales(self):
        print("Pesos iniciales de la capa oculta: \n",end='')
        print(*self.pesos_capaOculta)
        print("BIAS inicial de la capa oculta: \n",end='')
        print(*self.bias_capaOculta)
        print("Pesos iniciales de la capa de salida: \n",end='')
        print(*self.pesos_capaSalida)
        print("BIAS inicial de la capa de salida: \n",end='')
        print(*self.bias_capaSalida)
    
    def entrenar(self):
        for _ in range(self.epocas):
            #Forward Propagation
            hidden_layer_activation = np.dot(self.entradas,self.pesos_capaOculta)
            hidden_layer_activation += self.bias_capaOculta
            hidden_layer_output = sigmoid(hidden_layer_activation)
            
            output_layer_activation = np.dot(hidden_layer_output,self.pesos_capaSalida)
            output_layer_activation += self.bias_capaSalida

       
            predicted_output = sigmoid(output_layer_activation)
            self.salida = predicted_output

            #Backpropagation
            error = self.salidasEsperada - predicted_output
            d_predicted_output = error * sigmoid_derivative(predicted_output)


            error_hidden_layer = d_predicted_output.dot(self.pesos_capaSalida.T)
            d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)


            #Updating Weights and Biases
            self.pesos_capaSalida += hidden_layer_output.T.dot(d_predicted_output) * self.const_aprendizaje
            self.bias_capaSalida += np.sum(d_predicted_output) * self.const_aprendizaje
            self.pesos_capaOculta += self.entradas.T.dot(d_hidden_layer) * self.const_aprendizaje
            self.bias_capaOculta += np.sum(d_hidden_layer) * self.const_aprendizaje

            


    def mostrar_datos_finales(self):
        print("Peso finales de la capa oculta: \n",end='')
        print(self.pesos_capaOculta)
        print("BIAS finales de la capa oculta: \n",end='')
        print(self.bias_capaOculta)
        print("Peso finales de la capa salida:  \n",end='')
        print(self.pesos_capaSalida)
        print("BIAS finales de la capa salida: \n",end='')
        print(self.bias_capaSalida)
        print("\nSalidas de la red neuronal luego de 10,000 epocas: \n",end='   ')
        print(self.salida)

    def predecir(self):
            hidden_layer_activation = np.dot(self.prueba,self.pesos_capaOculta)
            hidden_layer_activation += self.bias_capaOculta
            hidden_layer_output = sigmoid(hidden_layer_activation)
            
            output_layer_activation = np.dot(hidden_layer_output,self.pesos_capaSalida)
            output_layer_activation += self.bias_capaSalida

       
            predicted_output = sigmoid(output_layer_activation)
            self.prediccion = predicted_output[0][0]
            return predicted_output[0][0]

    def Resultado(self):
        puntos = [0,0.15,0.30,0.45,0.60,0.75,0.90]
        menordist = 1000000
        respuesta = 0
        for i, item in enumerate(puntos):
            if abs(self.prediccion - item ) < menordist:
                menordist = abs(self.prediccion - item )
                respuesta = i
        return respuesta
            
        





iris_input = np.loadtxt("dataset.txt",delimiter=',',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
iris_output = np.loadtxt("dataset.txt",delimiter=',',usecols=[25])
prueba = [[0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0]]
aux = np.array(iris_output)
print(iris_input)
print(iris_output)

iris_output_matrix = []
b = []
for i in range(len(aux)):
    b.append(aux[i])
    iris_output_matrix.append(b)
    b = []



Iris = Backprogation(iris_input, iris_output_matrix,prueba)
Iris.mostar_datos_iniciales()
print("-------------------------------------")
Iris.entrenar()
print("-------------------------------------")
Iris.mostrar_datos_finales()


print(Iris.predecir())
print(Iris.Resultado())




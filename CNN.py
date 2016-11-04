import pandas as pd
import numpy as np

class CNN(object):
    def __init__(cnnFilter, stride, classes):
        self.filter = cnnFilter
        self.stride = stride
        self.classes = classes

    def weights(self, numOfInputs, numOfOutputs):
    def bias(self, numOfOutputs):
    def convolutionalLayer(self, train, weights):
    def relu(self, u):
    def MaxPoolingLayer2x2(self, y):
    def fullyConnectedLayer(self, y, weights):
        return tf.matmul(y, weights)'''to be changed'''
    def dropoutLayer(self, y, keepProbability):




"load files to arrays"
test = pd.read_csv("test.csv", header=None).as_matrix()
train = pd.read_csv("train.csv", header=None).as_matrix()
labelsTest = pd.read_csv("label_test.csv", header=None).as_matrix().flatten()#make labels a single list
labelsTrain = pd.read_csv("label_train.csv", header=None).as_matrix().flatten()

"initialize convolutional neural network with a filter and a stride"
cnnFilter = 25 #5x5
stride = 1 #step each time
classes = 10 #numbers from 0 to 9
cnn = CNN(cnnFilter, stride, classes)

"convolutional layer 1"
numOfInputs1 = len(train[0])#number of inputs
numOfOutputs1 = 32
weights1 = cnn.weights(numOfInputs1, numOfOutputs1)
biases1 = cnn.bias(numOfOutputs1)
conv1 = cnn.convolutionalLayer(train, weights1)
u1 = conv1 + biases1'''matrix addition must be implemented'''
y1 = cnn.relu(u1)

"pooling layer 1"
pool1 = cnn.MaxPoolingLayer2x2(y1)

"convolutional layer 2"
numOfInputs2 = numOfOutputs1
numOfOutputs2 = 64
weights2 = cnn.weights(numOfInputs2, numOfOutputs2)
biases2 = cnn.bias(numOfOutputs2)
conv2 = cnn.convolutionalLayer(pool1, weights2)
u2 = conv2 + biases2'''matrix addition must be implemented'''
y2 = cnn.relu(u2)

"pooling layer 2"
pool2 = cnn.MaxPoolingLayer2x2(y2)

"fully-connected layer"
weightsLast = cnn.weights(7*7*64, 1024)# image reduced to 7x7
biasLast = cnn.bias(1024)
fullyCon = cnn.fullyConnectedLayer(pool2, weightsLast)
uLast = fullyCon + biasLast'''matrix addition must be implemented'''
yLast = cnn.relu(uLast)

"dropout layer"
keepProbability = '''a number'''
dropout = cnn.dropoutLayer(yLast, keepProbability)

"softmax regression-like layer"
weightsEnd = weights(1024, classes)
biasEnd = bias(classes)

yConvolutional = tf.matmul(dropout, weightsEnd) + biasEnd'''must be changed'''

"---------------Training and Evaluation-------------------"

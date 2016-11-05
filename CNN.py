import tensorflow as tf
import numpy as np
import collections


class Dataset(object):
    def __init__(self, currentSet, labels, reshape = True):
        if (reshape):
            "supposing depth = 1 reshape to [numOfExamples, rows*columns]"
            currentSet = currentSet.reshape(currentSet.shape[0],
                currentSet.shape[1] * currentSet.shape[2])
        self.numOfExamples = currentSet.shape[0]
        self.set = currentSet
        self.labels = labels
        self.completedEpochs = 0
        self.epochIndex = 0

    def fetchBatch(self, batchSize):
        begin = self.epochIndex
        self.epochIndex += batchSize

        if (self.epochIndex > self.numOfExamples ):
            self.completedEpochs += 1

            "shuffle the rows of the dataset"
            permutation = np.arange(self.numOfExamples)
            np.random.shuffle(permutation)
            "apply shuffling"
            self.set = self.set[permutation]
            self.labels = self.labels[permutation]

            "begin new epoch"
            begin = 0
            self.epochIndex = batchSize #move index for the first time of this epoch

        stop = self.epochIndex

        return self.set[begin:stop], self.labels[begin:stop]

def getData(fileName, numOfImages):
  with open(fileName,"rb") as f:
    f.read(16)
    "read size of image x number of images"
    chunk = f.read(28 * 28 * numOfImages)
    "return a 1-dimensional array of integers and cast it to float32"
    data = np.frombuffer(chunk, dtype=np.uint8).astype(np.float32)
    "rescale from [0, 255] to [-0.5, 0.5]"
    data = (data - (255 / 2.0)) / 255
    "turn to a 4D tensor"
    data = data.reshape(numOfImages, 28, 28, 1)
    return data

def convertToOneHot(labels, numOfClasses):
    "one hot vector for e.g. number 3 = 0001000000"
  numOfLabels = labels.shape[0]
  offset = numpy.arange(numOfLabels) * numOfClasses
  oneHot = numpy.zeros((numOfLabels, numOfClasses))
  "find the exact positions to put 1"
  oneHot.flat[offset + labels.ravel()] = 1
  return oneHot

def getLabels(fileName, numOfImages, oneHot=False):
  with open(fileName,"rb") as f:
    f.read(8)
    chunk = f.read(numOfImages)
    "labels to int64 vector"
    labels = np.frombuffer(chunk, dtype=np.uint8).astype(np.int64)
    if oneHot:
        return convertToOneHot(labels,10)#10 is the number of classes
  return labels

def readData(oneHot=false):
    test = getData('t10k-images.idx3-ubyte', 10000)
    train = getData('train-images.idx3-ubyte', 60000)
    labelsTest = getLabels('t10k-labels.idx1-ubyte', 10000, oneHot)
    labelsTrain = getLabels('train-labels.idx1-ubyte', 60000, oneHot)

    validation = train[:5000]
    labelsValidation = labelsTrain[:5000]
    train= train[5000:]
    labelsTrain = labelsTrain[5000:]

    trainSet = Dataset(train, labelsTrain)
    validationSet = Dataset(validation, labelsValidation)
    testSet = Dataset(test, labelsTest)
    return collections.namedtuple('Datasets', ['trainSet','validationSet', 'testSet'])

"random weights with a small deviation that follow truncated normal distribution"
def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

"random positive bias to avoid dead neurons due to ReLU function"
def biases(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

"stride of 1 and zero padded"
def convolutionalLayer(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

"pooling over 2x2 blocks using max function"
def maxPoolingLayer2by2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def fullyConnectedLayer(x, W):
    return tf.matmul(x, W)

data = readData(True)#True is for one hot

session = tf.InteractiveSession()

"input ( 28 x 28 = 784 ) and output ( 10 possible one-hot outputs )"
x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 10])

"convolutional layer 1"
weights1 = weights([5, 5, 1, 32])#5x5 patch 1 input 32 outputs
biases1 = biases([32])

xReshaped = tf.reshape(x, [-1,28,28,1])

conv1 = convolutionalLayer(xReshaped, weights1)
u1 = conv1 + biases1
y1 = tf.nn.relu(u1)

"pooling layer 1"
pool1 = maxPoolingLayer2by2(y1)

"convolutional layer 2"
weights2 = weights([5, 5, 32, 64])#5x5 patch 32 inputs 64 outputs
biases2 = biases([64])

conv2 = convolutionalLayer(pool1, weights2)
u2 = conv2 = biases2
y2 = tf.nn.relu(u2)

"pooling layer 2"
pool2 = maxPoolingLayer2by2(y2)

"fully-connected layer"
pool2 = tf.reshape(pool2, [-1, 7*7*64])
"1024 neurons "
weightsLast = weights([7*7*64, 1024])# image reduced to 7x7
biasesLast = biases([1024])

fullyCon = fullyConnectedLayer(pool2,weightsLast)
uLast = fullyCon + biasesLast
yLast = tf.nn.relu(uLast)

"dropout layer"
keepProbability = tf.placeholder(tf.float32)
dropout = tf.nn.dropout(yLast, keepProbability)

"softmax regression-like layer"
weightsEnd = weights([1024, 10])
biasesEnd = biases([10])

yConvolutional = tf.matmul(dropout, weightsEnd) + biasesEnd

"---------------Training and Evaluation-------------------"
crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yConvolutional, y))
trainStep = tf.train.AdamOptimizer(1e-4).minimize(crossEntropy)
correctPrediction = tf.equal(tf.argmax(yConvolutional,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

session.run(tf.initialize_all_variables())

"define number of epochs"
for i in range(20000):
    batch = data.trainSet.fetchBatch(50)
    if i%100 == 0:
        trainAccuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1], keepProbability: 1.0})
        print("step %d, training accuracy %g"%(i, trainAccuracy))
    trainStep.run(feed_dict={x: batch[0], y: batch[1], keepProbability: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x: data.testSet.images,
    y: data.testSet.labels, keepProbability: 1.0}))

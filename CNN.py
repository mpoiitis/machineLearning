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

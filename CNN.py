import gzip
import numpy as np

IMAGE_SIZE = 28
NUM_CHANNELS = 1
def dataToArray(fileName, numOfImages):
  with open(fileName,"rb") as f:
    f.read(16)
    "read size of image x number of images"
    chunk = f.read(28 * 28 * numOfImages)
    "return a 1-dimensional array of integers and cast it to float32"
    data = np.frombuffer(chunk, dtype=np.uint8).astype(np.float32)
    "rescale from [0, 255] to [-0.5, 0.5]"
    data = (data - (255 / 2.0)) / 255
    "turn to a 4D tensor"
    data = data.reshape(numOfImages, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


def labelsToArray(fileName, numOfImages):
  with open(fileName,"rb") as f:
    f.read(8)
    chunk = f.read(numOfImages)
    "labels to int64 vector"
    labels = np.frombuffer(chunk, dtype=np.uint8).astype(np.int64)
  return labels

test = dataToArray('t10k-images.idx3-ubyte', 10000)
train = dataToArray('train-images.idx3-ubyte', 60000)
labelsTest = labelsToArray('t10k-labels.idx1-ubyte', 10000)
labelsTrain = labelsToArray('train-labels.idx1-ubyte', 60000)

import numpy as np
import time
from sklearn.decomposition import PCA

WITH_PCA =  True
INFO_PRES = 0.9

def getData(fileName, numOfImages):
  with open(fileName,"rb") as f:
    f.read(16)
    "read size of image x number of images"
    chunk = f.read(28 * 28 * numOfImages)
    "return a 1-dimensional array of integers and cast it to float32"
    data = np.frombuffer(chunk, dtype=np.uint8).astype(np.float32)
    "rescale from [0, 255] to [-0.5, 0.5]"
    data = (data - (255 / 2.0)) / 255
    "turn to a 2D array"
    data = data.reshape(numOfImages, 784)
    return data

def getLabels(fileName, numOfImages):
    with open(fileName,"rb") as f:
        f.read(8)
        chunk = f.read(numOfImages)
        "labels to int64 vector"
        labels = np.frombuffer(chunk, dtype=np.uint8).astype(np.int64)
    return labels

def readData():
    test = getData('t10k-images.idx3-ubyte', 10000)
    train = getData('train-images.idx3-ubyte', 60000)
    labelsTest = getLabels('t10k-labels.idx1-ubyte', 10000)
    labelsTrain = getLabels('train-labels.idx1-ubyte', 60000)
    if(WITH_PCA):
        #find number of attributes that we need to keep in order to preserve more than 90% of information
        pca = PCA()
        pca.fit(getData('train-images.idx3-ubyte', 60000))
        varianceList = np.cumsum(pca.explained_variance_ratio_)
        for index,variance in enumerate(varianceList):
            #the first value greater than 90% is the minimum number of attributes that fit our PCA
            if (variance > INFO_PRES):
                numOfAttrToKeep = index
                print("PCA with " + str(variance*100) + "% information preservation.")
                print("Attributes reduced from " + str(len(train[0])) + " to " + str(numOfAttrToKeep))
                break

        #true pca to reduce to the number of attributes in train and test set
        pca = PCA(n_components=numOfAttrToKeep)
        pca.fit(train)
        train = pca.transform(train)# now train set has undergone PCA
        #pca.fit(test)
        test = pca.transform(test)

    data = {'train': {'data': train,
                      'labels': labelsTrain},
            'test': {'data': test,
                     'labels': labelsTest}}
    return data

data = readData()

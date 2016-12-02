import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.decomposition import PCA

KERNEL_TYPE = "linear"
WITH_PCA = True

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
        #change labels due to the binary nature of the svm
        labels = [-1 if(label%2 == 0) else 1 for label in labels]
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
            if (variance > 0.9):
                numOfAttrToKeep = index
                print("PCA with " + str(variance*100) + "% information preservation.")
                print("Attributes reduced from " + str(len(train[0])) + " to " + str(numOfAttrToKeep))
                break

        #true pca to reduce to the number of attributes in train and test set
        pca = PCA(n_components=numOfAttrToKeep)
        pca.fit(train)
        train = pca.transform(train)# now train set has undergone PCA
        pca.fit(test)
        test = pca.transform(test)

    data = {'train': {'data': train,
                      'labels': labelsTrain},
            'test': {'data': test,
                     'labels': labelsTest}}
    return data

data = readData()

#Classifier, default c=1.0, default kernel=rbf, default degree=3, default gamma=1/n_features
classifier = LinearSVC()

length = len(data['train']['data'])
#Get training time
startTime = time.time()
classifier.fit(data['train']['data'][:length], data['train']['labels'][:length])
endTime = time.time()
# Get confusion matrix
predicted = classifier.predict(data['test']['data'])
print("Confusion matrix:\n%s" %
         metrics.confusion_matrix(data['test']['labels'],
                                  predicted))
print("Accuracy: %0.4f" % metrics.accuracy_score(data['test']['labels'],
                                                    predicted))
print("Time needed for training: " + str((endTime - startTime)/60.) + " minutes")
print("Kernel type: " + str(KERNEL_TYPE))

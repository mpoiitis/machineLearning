import pandas as pd
import numpy as np

test = pd.read_csv("test.csv", header=None).as_matrix()
train = pd.read_csv("train.csv", header=None).as_matrix()
labelsTest = pd.read_csv("label_test.csv", header=None).as_matrix().flatten()#make labels a single list
labelsTrain = pd.read_csv("label_train.csv", header=None).as_matrix().flatten()

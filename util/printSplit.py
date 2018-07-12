import os
import numpy as np

base_dir = '/data2/nathan/embryo/data/'
plist = []
for f in os.listdir(base_dir):
   if f[:29] not in plist:
      plist.append(f[:29])
np.random.seed(69)
split = np.random.permutation(len(plist))
testSplit = split[-10:]
valSplit = split[-20:-10]
trainSplit = split[:-20]
testSplit = [plist[i] for i in testSplit]
trainSplit = [plist[i] for i in trainSplit]
valSplit = [plist[i] for i in valSplit]
print(testSplit)
print(valSplit)
print(trainSplit)

import os
import numpy as np
import cv2
import cPickle as pkl
import json

def patientSplit():
   base_dir = '/data2/nathan/embryo/data/'
   plist = []
   for f in os.listdir(base_dir):
      if f[:29] not in plist:
         plist.append(f[:29])
   np.random.seed(69)
   split = np.random.permutation(len(plist))
   print(len(split))
   testSplit = [plist[i] for i in split[-10:]]
   valSplit = [plist[i] for i in split[-20:-10]]
   trainSplit = [plist[i] for i in split[:-20]]
   ftest, fval, ftrain = [], [], []
   for f in os.listdir(base_dir):
      if f[:29] in testSplit:
         if 'feat' in f:
            ftest.append(f[:-8])
      elif f[:29] in valSplit:
         if 'feat' in f:
            fval.append(f[:-8])
      elif f[:29] in trainSplit:
         if 'feat' in f:
            ftrain.append(f[:-8])
   with open('testSplit.json', 'wb') as of:
      json.dump(ftest, of)
   with open('valSplit.json', 'wb') as of:
      json.dump(fval, of)
   with open('trainSplit.json', 'wb') as of:
      json.dump(ftrain, of)

   X_train = np.load(base_dir + ftrain[0] + 'feat.npy')
   Y_train = np.load(base_dir + ftrain[0] + 'labelRaw.npy')
   for f in ftrain[1:]:
      print f
      X_imm = np.load(base_dir + f + 'feat.npy')
      Y_imm = np.load(base_dir + f + 'labelRaw.npy')
      X_train = np.vstack([X_train, X_imm])
      Y_train = np.vstack([Y_train, Y_imm])
   X_val = np.load(base_dir + fval[0] + 'feat.npy')
   Y_val = np.load(base_dir + fval[0] + 'labelRaw.npy')
   for f in fval[1:]:
      print f
      X_imm = np.load(base_dir + f + 'feat.npy')
      Y_imm = np.load(base_dir + f + 'labelRaw.npy')
      X_val = np.vstack([X_val, X_imm])
      Y_val = np.vstack([Y_val, Y_imm])
   X_test = np.load(base_dir + ftest[0] + 'feat.npy')
   Y_test = np.load(base_dir + ftest[0] + 'labelRaw.npy')
   for f in ftest[1:]:
      print f
      X_imm = np.load(base_dir + f + 'feat.npy')
      Y_imm = np.load(base_dir + f + 'labelRaw.npy')
      X_test = np.vstack([X_test, X_imm])
      Y_test = np.vstack([Y_test, Y_imm])
   np.save(open('/data2/nathan/embryo/train/featRaw.npy', 'wb'), X_train)
   np.save(open('/data2/nathan/embryo/train/labelRaw.npy', 'wb'), Y_train)
   np.save(open('/data2/nathan/embryo/test/featRaw.npy', 'wb'), X_test)
   np.save(open('/data2/nathan/embryo/test/labelRaw.npy', 'wb'), Y_test)
   np.save(open('/data2/nathan/embryo/val/featRaw.npy', 'wb'), X_val)
   np.save(open('/data2/nathan/embryo/val/labelRaw.npy', 'wb'), Y_val)

if __name__ == '__main__':
   patientSplit()

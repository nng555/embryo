import os
import numpy as np
import cv2
import cPickle as pkl

def patientSplit():
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
   X_train = np.load(base_dir + plist[trainSplit[0]] + 'feat.npy')
   Y_train = np.load(base_dir + plist[trainSplit[0]] + 'label.npy')
   for index in trainSplit[1:]:
      X_imm = np.load(base_dir + plist[index] + 'feat.npy')
      Y_imm = np.load(base_dir + plist[index] + 'label.npy')
      X_train = np.vstack([X_train, X_imm])
      Y_train = np.vstack([Y_train, Y_imm])
   X_test = np.load(base_dir + plist[testSplit[0]] + 'feat.npy')
   Y_test = np.load(base_dir + plist[testSplit[0]] + 'label.npy')
   for index in testSplit[1:]:
      X_imm = np.load(base_dir + plist[index] + 'feat.npy')
      Y_imm = np.load(base_dir + plist[index] + 'label.npy')
      X_test = np.vstack([X_test, X_imm])
      Y_test = np.vstack([Y_test, Y_imm])
   X_val = np.load(base_dir + plist[valSplit[0]] + 'feat.npy')
   Y_val = np.load(base_dir + plist[valSplit[0]] + 'label.npy')
   for index in valSplit[1:]:
      X_imm = np.load(base_dir + plist[index] + 'feat.npy')
      Y_imm = np.load(base_dir + plist[index] + 'label.npy')
      X_val = np.vstack([X_val, X_imm])
      Y_val = np.vstack([Y_val, Y_imm])
   np.save(open('/data2/nathan/embryo/train/featRaw.npy', 'wb'), X_train)
   np.save(open('/data2/nathan/embryo/train/labelRaw.npy', 'wb'), Y_train)
   np.save(open('/data2/nathan/embryo/test/featRaw.npy', 'wb'), X_test)
   np.save(open('/data2/nathan/embryo/test/labelRaw.npy', 'wb'), Y_test)
   np.save(open('/data2/nathan/embryo/val/featRaw.npy', 'wb'), X_val)
   np.save(open('/data2/nathan/embryo/val/labelRaw.npy', 'wb'), Y_val)

if __name__ == '__main__':
   patientSplit()

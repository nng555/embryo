import numpy as np
import os
from sklearn.model_selection import train_test_split
import json

def combineMat():
   base_dir = '/data2/nathan/embryo/'
   numExamples = 0
   for i in range(34):
      y = np.load(base_dir + 'data/labelMat' + str(i) + '.npy')
      numExamples += len(y)

   order = range(34)
   np.random.seed(69)
   np.random.shuffle(order)


   X_train = np.load(base_dir + 'data/featMat' + str(order[0]) + '.npy')
   Y_train = np.load(base_dir + 'data/labelMat' + str(order[0]) + '.npy')
   X_val = np.load(base_dir + 'data/featMat' + str(order[1]) + '.npy')
   Y_val = np.load(base_dir + 'data/labelMat' + str(order[1]) + '.npy')
   X_test = np.load(base_dir + 'data/featMat' + str(order[2]) + '.npy')
   Y_test = np.load(base_dir + 'data/labelMat' + str(order[2]) + '.npy')
   trainPat = np.unique(X_train[:,2])
   valPat = np.unique(X_val[:,2])
   testPat = np.unique(X_test[:,2])
   for mat in order[3:]:
      X_imm = np.load(base_dir + 'data/featMat' + str(mat) + '.npy')
      Y_imm = np.load(base_dir + 'data/labelMat' + str(mat) + '.npy')
      if len(X_train) < 0.8 * numExamples:
         X_train = np.vstack([X_train, X_imm])
         Y_train = np.vstack([Y_train, Y_imm])
         trainPat = np.hstack([np.unique(X_imm[:,2]), trainPat])
      elif len(X_val) < 0.08 * numExamples:
         X_val = np.vstack([X_val, X_imm])
         Y_val = np.vstack([Y_val, Y_imm])
         valPat = np.hstack([np.unique(X_imm[:,2]), valPat])
      else:
         X_test = np.vstack([X_test, X_imm])
         Y_test = np.vstack([Y_test, Y_imm])
         testPat = np.hstack([np.unique(X_imm[:,2]), testPat])

   np.save(open(base_dir + 'train/feat.npy', 'wb'), X_train)
   np.save(open(base_dir + 'train/label.npy', 'wb'), Y_train)
   np.save(open(base_dir + 'test/feat.npy', 'wb'), X_test)
   np.save(open(base_dir + 'test/label.npy', 'wb'), Y_test)
   np.save(open(base_dir + 'val/feat.npy', 'wb'), X_val)
   np.save(open(base_dir + 'val/label.npy', 'wb'), Y_val)
   np.save(open(base_dir + 'train/patients.npy', 'wb'), trainPat)
   np.save(open(base_dir + 'val/patients.npy', 'wb'), valPat)
   np.save(open(base_dir + 'test/patients.npy', 'wb'), testPat)

if __name__ == "__main__":
   combineMat()

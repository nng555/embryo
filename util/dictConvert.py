import os
import cPickle as pkl
import cv2
import numpy as np

def dictConvert(size):
   base_dir = '/data2/nathan/embryo/data/'
   for i in range(34):
      X = []
      Y = []
      feat = pkl.load(open(base_dir + 'patientFeat' + str(i) + '.pkl'))
      label = pkl.load(open(base_dir + 'patientLabel' + str(i) + '.pkl'))
      for k in feat.keys():
         for idx in range(len(feat[k])):
            #gray = cv2.cvtColor(feat[k][idx][0], cv2.COLOR_BGR2GRAY)
            res = cv2.resize(feat[k][idx][0], (size, size))
            X.append([res, feat[k][idx][1], k])
            Y.append(label[k][idx])
      np.save(open(base_dir + 'featMat' + str(i) + '.npy', 'wb'), np.asarray(X))
      np.save(open(base_dir + 'labelMat' + str(i) + '.npy', 'wb'), np.asarray(Y))

if __name__ == '__main__':
   size = int(raw_input("Size: "))
   dictConvert(size)

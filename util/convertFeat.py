import numpy as np
import os
import cv2

def convertFeat():
   '''
   X_val = np.load('/data2/nathan/embryo/val/featRaw.npy')
   for ex in X_val:
      ex[0] = cv2.resize(ex[0], (224, 224))
   np.save(open('/data2/nathan/embryo/val/feat224.npy', 'wb'), X_val)
   '''
   X_train = np.load('/data2/nathan/embryo/train/featRaw.npy')
   for ex in X_train:
      ex[0] = cv2.resize(ex[0], (224, 224))
   np.save(open('/data2/nathan/embryo/train/feat224.npy', 'wb'), X_train)
   X_test = np.load('/data2/nathan/embryo/test/featRaw.npy')
   for ex in X_test:
      ex[0] = cv2.resize(ex[0], (224, 224))
   np.save(open('/data2/nathan/embryo/test/feat224.npy', 'wb'), X_test)

if __name__ == '__main__':
   convertFeat()

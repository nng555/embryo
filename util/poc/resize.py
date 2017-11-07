import numpy as np
import cv2

from PIL import Image, ImageFilter

def resizeFrames(size):
   X_train = np.load('/data2/nathan/embryo/train/featOneHot.npy')
   X_val = np.load('/data2/nathan/embryo/val/featOneHot.npy')
   X_test = np.load('/data2/nathan/embryo/test/featOneHot.npy')
   dsets = [X_train, X_val, X_test]
   frames = [[], [], []]
   for i in range(len(dsets)):
      for exm in dsets[i]:
         #gray = cv2.cvtColor(exm[0], cv2.COLOR_BGR2GRAY)
         exm[0] = cv2.resize(exm[0], (size, size))
         frames[i].append(exm)
   frames = np.asarray(frames)
   np.save(open('/data2/nathan/embryo/train/feat' + str(size) + '.npy', 'wb'), frames[0])
   np.save(open('/data2/nathan/embryo/val/feat' + str(size) + '.npy', 'wb'), frames[1])
   np.save(open('/data2/nathan/embryo/test/feat' + str(size) + '.npy', 'wb'), frames[2])

if __name__ == '__main__':
   size = int(raw_input("Size: "))
   resizeFrames(size)

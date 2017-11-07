import numpy as np
import cPickle as pkl
import json

from sklearn.metrics import accuracy_score, log_loss

def trainMode():
   X_train = np.load('/data2/nathan/embryo/train/feat.npy')
   Y_train = np.load('/data2/nathan/embryo/train/label.npy')

   frameCount = {}
   for i in range(len(X_train)):
      frame = round(X_train[i][1], 1)
      if frame not in frameCount:
         frameCount[frame] = []
      frameCount[frame].append(np.where(Y_train[i] == 1)[0][0])
   frameMode = {}
   for k, v in frameCount.iteritems():
      val, counts = np.unique(v, return_counts=True)
      frameMode[k] = val[np.argmax(counts)]

   X_test = np.load('/data2/nathan/embryo/test/feat.npy')
   Y_test = np.load('/data2/nathan/embryo/test/label.npy')

   pred = []
   predHot = []
   for exm in X_test:
      frame = round(exm[1], 1)
      hot = [0 for i in range(7)]
      if frame in frameMode:
         hot[frameMode[frame]] = 1
         pred.append(frameMode[frame])
         predHot.append(hot)
      else:
         time = frame + 0.1
         while(time not in frameMode):
            time = time + 0.1
         hot[frameMode[time]] = 1
         pred.append(frameMode[time])
         predHot.append(hot)

   Y_testi = []
   for label in Y_test:
      Y_testi.append(np.where(label == 1)[0][0])

   Y_testi = np.asarray(Y_testi)
   pred = np.asarray(pred)

   print(accuracy_score(Y_testi, pred))
   print(log_loss(Y_testi, predHot))

   pred = []
   predHot = []
   for exm in X_train:
      frame = round(exm[1], 1)
      hot = [0 for i in range(7)]
      if frame in frameMode:
         hot[frameMode[frame]] = 1
         pred.append(frameMode[frame])
         predHot.append(hot)
      else:
         time = frame + 0.1
         while(time not in frameMode):
            time = time + 0.1
         hot[frameMode[frame]] = 1
         pred.append(frameMode[time])
         predHot.append(hot)
   Y_traini = []
   for label in Y_train:
      Y_traini.append(np.where(label == 1)[0][0])

   Y_traini = np.asarray(Y_traini)
   pred = np.asarray(pred)

   print(accuracy_score(Y_traini, pred))
   print(log_loss(Y_traini, predHot))

if __name__ == '__main__':
   trainMode()

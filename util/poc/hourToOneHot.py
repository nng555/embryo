import numpy as np
from collections import defaultdict

def buildHoursList():
   X_train = np.load('/data2/nathan/embryo/train/feat.npy')
   X_test = np.load('/data2/nathan/embryo/test/feat.npy')
   X_val = np.load('/data2/nathan/embryo/val/feat.npy')
   oneHot = defaultdict(int)
   for dset in [X_train, X_test, X_val]:
      for val in dset[:,1]:
         oneHot[np.round(val, 1)] += 1
   hours = np.sort(oneHot.keys()).tolist()
   return hours

def hourToOneHot(fname, hours):
   X = np.load('/data2/nathan/embryo/' + fname + '/feat.npy')
   indices = np.array([hours.index(np.round(exm[1], 1)) for exm in X])
   zeroes = np.zeros((len(X), len(hours)))
   zeroes[np.arange(len(X)), indices] = 1
   res = []
   for i in range(len(X)):
      res.append([X[i][0], zeroes[i]])
   np.save(open('/data2/nathan/embryo/' + fname + '/featOneHot.npy', 'wb'), res)

if __name__ == '__main__':
   hours = buildHoursList()
   for fname in ('train', 'test', 'val'):
      hourToOneHot(fname, hours)

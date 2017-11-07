import numpy as np
import cPickle as pkl

def dataSplit():
   feat = pkl.load(open('/data2/nathan/embryo/patientFeat.pkl'))
   label = pkl.load(open('/data2/nathan/embryo/patientLabelBin.pkl'))

   testPatient = 2

   split = [[] for i in range(6)]

   counter = 0
   for (p, w) in feat.keys():
      if p == testPatient:
         split[4].extend(feat[p, w])
         split[5].extend(label[p, w])
      elif (p, w) in [(1, 3), (0, 7), (3, 8)]:
         split[2].extend(feat[p, w])
         split[3].extend(label[p, w])
      else:
         split[0].extend(feat[p, w])
         split[1].extend(label[p, w])

   split = [np.asarray(arr) for arr in split]

   np.save(open('/data2/nathan/embryo/train/feat.npy', 'wb'), split[0])
   np.save(open('/data2/nathan/embryo/train/label.npy', 'wb'), split[1])
   np.save(open('/data2/nathan/embryo/val/feat.npy', 'wb'), split[2])
   np.save(open('/data2/nathan/embryo/val/label.npy', 'wb'), split[3])
   np.save(open('/data2/nathan/embryo/test/feat.npy', 'wb'), split[4])
   np.save(open('/data2/nathan/embryo/test/label.npy', 'wb'), split[5])


if __name__ == '__main__':
   dataSplit()

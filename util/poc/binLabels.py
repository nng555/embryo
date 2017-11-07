import numpy as np
import cPickle as pkl
from collections import defaultdict

def binLabels(bins):
   Y_patient = pkl.load(open('/data2/nathan/embryo/patientLabel.pkl'))

   Y_bin = defaultdict(list)
   for ((patient, well), labels) in Y_patient.iteritems():
      for label in labels:
         res = [0 for j in range(len(bins))]
         index = label.index(1)
         for k in range(len(bins)):
            if index in bins[k]:
               res[k] = 1
               break
         Y_bin[patient, well].append(res)

   with open('/data2/nathan/embryo/patientLabelBin.pkl', 'wb') as of:
      pkl.dump(Y_bin, of, -1)

if __name__ == '__main__':
   bins = [[0, 1],
           [2, 3],
           [4, 5, 6, 7],
           [8],
           [9],
           [10, 11],
           [12, 13, 14]]
   binLabels(bins)

import os
import numpy as np
import cPickle as pkl
import json

def discardSplit():
   base_dir = '/data2/nathan/embryo/data/'
   dlist = []
   for f in os.listdir(base_dir):
      if 'discard' in f and 'feat' in f:
         dlist.append(f[:-8])
   X_disc = np.load(base_dir + dlist[0] + 'feat.npy')
   Y_disc = np.load(base_dir + dlist[0] + 'labelRaw.npy')
   for f in dlist[1:]:
      X_imm = np.load(base_dir + f + 'feat.npy')
      Y_imm = np.load(base_dir + f + 'labelRaw.npy')
      X_disc = np.vstack([X_disc, X_imm])
      Y_disc = np.vstack([Y_disc, Y_imm])

   np.save(open('/data2/nathan/embryo/discard/featRaw.npy', 'wb'), X_disc)
   np.save(open('/data2/nathan/embryo/discard/labelRaw.npy', 'wb'), Y_disc)

if __name__ == '__main__':
   discardSplit()

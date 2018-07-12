import numpy as np

for dset in ['discard']:
   feat = np.load('/data2/nathan/embryo/' + dset + '/featRaw.npy')
   label = np.load('/data2/nathan/embryo/' + dset + '/labelRaw.npy')
   featCut = []
   labelCut = []
   for i in range(len(feat)):
      if feat[i][1] < 60.0:
         featCut.append(feat[i])
         lcut = label[i][:6]
         if 1 not in lcut:
            lcut[-1] = 1
         labelCut.append(lcut)
   featCut = np.asarray(featCut)
   labelCut = np.asarray(labelCut)
   np.save(open('/data2/nathan/embryo/' + dset+ '/featCut.npy', 'wb'), featCut)
   np.save(open('/data2/nathan/embryo/' + dset+ '/labelCut.npy', 'wb'), labelCut)

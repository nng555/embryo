import numpy as np

def oneHotToIndex():
   for dset in ['train', 'val', 'test', 'discard']:
      Y = np.load('/data2/nathan/embryo/' + dset + '/labelRaw.npy')
      res_Y = [np.argmax(lab) for lab in Y]
      np.save(open('/data2/nathan/embryo/' + dset + '/labelRawIndex.npy', 'wb'), res_Y)

if __name__ == '__main__':
   oneHotToIndex()

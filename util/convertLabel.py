import numpy as np

def convertLabel(dset):
   label = np.load('/data2/nathan/embryo/' + dset + '/labelRaw.npy')
   for ex in label:
      index = np.where(ex==1)[0][0]
      for ordi in range(index):
         ex[ordi] = 1
   np.save(open('/data2/nathan/embryo/' + dset + '/labelOrdinal.npy', 'wb'), label)

if __name__ == "__main__":
   convertLabel('train')
   convertLabel('test')
   convertLabel('val')

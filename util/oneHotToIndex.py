import numpy as np

def oneHotToIndex():
   Y_train = np.load('/data2/nathan/embryo/train/labelRaw.npy')
   Y_test = np.load('/data2/nathan/embryo/test/labelRaw.npy')
   Y_val = np.load('/data2/nathan/embryo/val/labelRaw.npy')
   #Y_toy = np.load('/data2/nathan/embryo/toy/label.npy')
   res_train = [np.argmax(lab) for lab in Y_train]
   res_test = [np.argmax(lab) for lab in Y_test]
   res_val = [np.argmax(lab) for lab in Y_val]
   #res_toy = [np.argmax(lab) for lab in Y_toy]
   np.save(open('/data2/nathan/embryo/train/labelIndex.npy', 'wb'), res_train)
   np.save(open('/data2/nathan/embryo/test/labelIndex.npy', 'wb'), res_test)
   np.save(open('/data2/nathan/embryo/val/labelIndex.npy', 'wb'), res_val)
   #np.save(open('/data2/nathan/embryo/toy/labelIndex.npy', 'wb'), res_toy)

if __name__ == '__main__':
   oneHotToIndex()

import numpy as np

def genToy():
   X_train = np.load('/data2/nathan/embryo/train/featFlat.npy')
   Y_train = np.load('/data2/nathan/embryo/train/label.npy')
   res_x = []
   res_y = []
   for i in range(len(X_train)):
      if i % 50 == 0:
         res_x.append(X_train[i])
         res_y.append(Y_train[i])
   res_x = np.asarray(res_x)
   res_y = np.asarray(res_y)
   np.save(open('/data2/nathan/embryo/toy/featFlat.npy', 'wb'), res_x)
   np.save(open('/data2/nathan/embryo/toy/label.npy', 'wb'), res_y)



if __name__ == '__main__':
   genToy()

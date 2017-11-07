import numpy as np

def flattenFeat():
   X_train = np.load('/data2/nathan/embryo/train/feat32.npy')
   X_test = np.load('/data2/nathan/embryo/test/feat32.npy')
   X_val = np.load('/data2/nathan/embryo/val/feat32.npy')
   res_train = [exm[0].flatten()[0::3] for exm in X_train]
   res_test = [exm[0].flatten()[0::3] for exm in X_test]
   res_val = [exm[0].flatten()[0::3] for exm in X_val]
   np.save(open('/data2/nathan/embryo/train/featFlat.npy', 'wb'), res_train)
   np.save(open('/data2/nathan/embryo/test/featFlat.npy', 'wb'), res_test)
   np.save(open('/data2/nathan/embryo/val/featFlat.npy', 'wb'), res_val)

if __name__ == '__main__':
   flattenFeat()

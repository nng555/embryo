import numpy as np
import cv2

def flattenFeat():
   X_train = np.load('/data2/nathan/embryo/train/featRaw.npy')
   X_test = np.load('/data2/nathan/embryo/test/featRaw.npy')
   X_val = np.load('/data2/nathan/embryo/val/featRaw.npy')
   res_train = [cv2.resize(exm[0], (32, 32)).flatten()[0::3] for exm in X_train]
   res_test = [cv2.resize(exm[0], (32, 32)).flatten()[0::3] for exm in X_test]
   res_val = [cv2.resize(exm[0], (32, 32)).flatten()[0::3] for exm in X_val]
   np.save(open('/data2/nathan/embryo/train/featFlat.npy', 'wb'), res_train)
   np.save(open('/data2/nathan/embryo/test/featFlat.npy', 'wb'), res_test)
   np.save(open('/data2/nathan/embryo/val/featFlat.npy', 'wb'), res_val)

if __name__ == '__main__':
   flattenFeat()

import numpy as np
import argparse
import cPickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def regTrain(size, l2):

   X_train = np.load('/data2/nathan/embryo/train/featFlat.npy')
   Y_train = np.load('/data2/nathan/embryo/train/labelIndex.npy')

   reg = LogisticRegression(C=l2, multi_class='ovr')
   reg.fit(X_train, Y_train)

   with open('/data2/nathan/embryo/reg' + str(l2) + '.pkl', 'wb') as of:
      pkl.dump(reg, of, -1)

   X_val = np.load('/data2/nathan/embryo/test/featFlat.npy')
   Y_val = np.load('/data2/nathan/embryo/test/labelIndex.npy')
   Y_hot = np.load('/data2/nathan/embryo/test/label.npy')

   pred = reg.predict(X_val)
   print(accuracy_score(Y_val, pred))

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('-s', action='store', dest='size', type=int,
         help='the size of the training images')
   parser.add_argument('-l', action='store', dest='l2', type=float,
         help='the l2 loss constant')
   args = vars(parser.parse_args())
   regTrain(args['size'], args['l2'])

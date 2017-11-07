import os
import csv
import numpy as np
import cv2
import cPickle as pkl
from collections import defaultdict

from PIL import Image, ImageFilter

def splitFrames():

   # load in the csv and retrieve the header indices
   reader = csv.reader(open('../labels.csv', 'rU'))
   header = reader.next()
   df = [row for row in reader]
   indict = {}
   for i in range(len(header)):
      indict[header[i]] = i
   indict['Patient'] = 0

   # load rows into dictionary if they are invideotestdataset
   pdict = {}
   for row in df:
      pind = indict['Patient']
      wind = indict['Well']
      if row[indict['videointestdataset']] == '1':
         if int(row[pind]) not in pdict:
            pdict[int(row[pind])] = {}
         pdict[int(row[pind])][int(row[wind])] = row

   #X = [[] for i in range(4)]
   #Y = [[] for i in range(4)]
   X = defaultdict(list)
   Y = defaultdict(list)

   # load model
   clf = pkl.load(open('/home/nathan/embryo/timeClf.pkl', 'rb'))

   # check each video
   for f in os.listdir('../samplepatients'):
      print(f)
      patientNum = int(f[7])
      wells = [int(well) for well in f.split('wells_')[1].split('_video')[0].split('_')]
      for i in range(len(wells)):
         data =  pdict[patientNum][wells[i]]
         tpnfInd = indict['tPNf']
         thbInd = indict['tHB']
         wdInd = indict['Well Description']

         times = [0] + [float(time) if time != '' else -1 for time in data[tpnfInd:thbInd+1]]

         cap = cv2.VideoCapture('/home/nathan/embryo/samplepatients/' + f)
         lastFrame = 0
         while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret or np.sum((frame-lastFrame)**2) < 1e7:
               break
            lastFrame = frame
            frame = frame[:,i*500:(i+1)*500]
            gray = cv2.cvtColor(frame[475:490,449:456], cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
            dig1 = clf.predict([thresh.reshape(105)])[0]
            if(dig1 != 1):
               dig1 = 0

            gray = cv2.cvtColor(frame[475:490,456:463], cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
            dig2 = clf.predict([thresh.reshape(105)])[0]

            gray = cv2.cvtColor(frame[475:490,463:470], cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
            dig3 = clf.predict([thresh.reshape(105)])[0]

            gray = cv2.cvtColor(frame[475:490,473:480], cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
            dig4 = clf.predict([thresh.reshape(105)])[0]

            hr = 100*dig1 + dig2*10 + dig3 + dig4/10.0

            tpnfInd = indict['tPNf']
            thbInd = indict['tHB']
            wdInd = indict['Well Description']

            times = [0] + [float(time) if time != '' else -1 for time in data[tpnfInd:thbInd+1]]
            timeRes = [0 for j in range(len(times))]
            for j in range(len(times) - 1):
               if times[j] != -1:
                  if times[j] <= hr and hr < times[j+1]:
                     timeRes[j] = 1
                     break
            if hr >= times[-1] and times[-1] != -1:
               timeRes[-1] = 1

            if 1 not in timeRes:
               continue

            wdRes = [0, 0, 0, 0]
            if data[wdInd] == '' or data[wdInd] == 'NO RESULT':
               wdRes[0] = 1
            elif data[wdInd] == 'NL':
               wdRes[1] = 1
            elif data[wdInd] == 'ABN':
               wdRes[2] = 1
            elif data[wdInd] == 'NC' or data[wdInd] == 'NONCONCURRENT':
               wdRes[3] = 1

            X[patientNum - 1, wells[i]].append([frame, hr])
            Y[patientNum - 1, wells[i]].append(timeRes)
            #X[patientNum - 1].append([frame, hr, patientNum, wells[i]])
            #Y[patientNum - 1].append(timeRes)

   '''
   X = np.asarray(X)
   Y = np.asarray(Y)
   with open('/data2/nathan/embryo/patientFeat.npy', 'wb') as of:
      np.save(of, X)
   with open('/data2/nathan/embryo/patientLabel.npy', 'wb') as of:
      np.save(of, Y)
   '''

   with open('/data2/nathan/embryo/patientFeat.pkl', 'wb') as of:
      pkl.dump(X, of, -1)
   with open('/data2/nathan/embryo/patientLabel.pkl', 'wb') as of:
      pkl.dump(Y, of, -1)

if __name__ == "__main__":
   splitFrames()

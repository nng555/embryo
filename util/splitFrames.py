import os
import csv
import codecs
import numpy as np
import cv2
import cPickle as pkl
from collections import defaultdict

from PIL import Image, ImageFilter

def splitFrames():

   # load in csv and retrieve header indices
   reader = csv.reader(open("/data2/nathan/embryo/filteredLabels.csv", "rU"), delimiter='\t')
   header = reader.next()
   print(header)
   df = [row for row in reader]
   indict = {}
   for i in range(len(header)):
      indict[header[i]] = i

   # set index variables
   pind = 0
   wind = indict['Well']
   tpnfInd = indict['tPNf']
   thbInd = indict['tHB']

   # load rows indexed on slideID and well
   pdict = {}
   for row in df:
      if row[wind] == '':
         break
      if row[pind] not in pdict:
         pdict[row[pind]] = {}
      pdict[row[pind]][int(row[wind])] = row

   loaded = {}

   # load model
   clf = pkl.load(open('/home/nathan/embryo/timeClf.pkl', 'rb'))

   # check each video
   for f in os.listdir('/data2/nathan/embryo/videos'):
      slide = f.split('_wells_')[0]
      if slide not in pdict:
         continue
      print(f)
      wells = [int(well) for well in f.split('_wells_')[1].split('_')[:-1]]
      for i in range(len(wells)):
         if wells[i] not in pdict[slide]:
            continue
         if (slide, wells[i]) in loaded:
            continue
         loaded[slide, wells[i]] = 1
         X = []
         Y = []
         data = pdict[slide][wells[i]]

         times = [0] + [float(time) if time != '' else -1 for time in data[tpnfInd:thbInd+1]]

         cap = cv2.VideoCapture('/data2/nathan/embryo/videos/' + f)
         lastFrame = 0
         while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret or np.sum((frame-lastFrame)**2) < 1e7:
               break
            lastFrame = frame
            if len(wells) == 4:
               frame = frame[(i%2)*250:((i%2)+1)*250,(i/2)*250:((i/2)+1)*250]
            else:
               frame = frame[:,i*500:(i+1)*500]
            fLen = len(frame)
            gray = cv2.cvtColor(frame[fLen-25:fLen-10,fLen-51:fLen-44], cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
            dig1 = clf.predict([thresh.reshape(105)])[0]
            if(dig1 != 1):
               dig1 = 0

            gray = cv2.cvtColor(frame[fLen-25:fLen-10,fLen-44:fLen-37], cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
            dig2 = clf.predict([thresh.reshape(105)])[0]

            gray = cv2.cvtColor(frame[fLen-25:fLen-10,fLen-37:fLen-30], cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
            dig3 = clf.predict([thresh.reshape(105)])[0]

            gray = cv2.cvtColor(frame[fLen-25:fLen-10,fLen-27:fLen-20], cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
            dig4 = clf.predict([thresh.reshape(105)])[0]

            hr = 100*dig1 + dig2*10 + dig3 + dig4/10.0

            timeRes = [0 for j in range(len(times))]
            last = -1
            for j in range(len(times)):
               if times[j] <= hr and times[j] != -1:
                  last = j
            if last != -1:
               timeRes[last] = 1
            else:
               continue

            frame = cv2.resize(frame, (224, 224))
            X.append([frame, hr])
            Y.append(timeRes)

         X = np.asarray(X)
         Y = np.asarray(Y)
         '''
         if ('patient' + str(slide) + 'labelRaw.npy') in  os.listdir('/data2/nathan/embryo/data'):
            X_imm = np.load('/data2/nathan/embryo/data/patient' + str(slide) + 'feat.npy')
            Y_imm = np.load('/data2/nathan/embryo/data/patient' + str(slide) + 'labelRaw.npy')
            X = np.vstack([X_imm, X])
            Y = np.vstack([Y_imm, Y])
         '''
         np.save(open('/data2/nathan/embryo/data/patient' + str(slide) + 'well' + str(wells[i]) + 'feat.npy', 'wb'), X)
         np.save(open('/data2/nathan/embryo/data/patient' + str(slide) + 'well' + str(wells[i]) + 'labelRaw.npy', 'wb'), Y)

if __name__ == "__main__":
   splitFrames()

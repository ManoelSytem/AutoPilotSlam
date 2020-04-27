import cv2
import numpy as np


class Extractor(object):
     GX = 16//2
     GY = 12//2

     def __init__(self):
        self.orb = cv2.ORB_create(100)
        self.bf  =  cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
     
     def extractor(self,img):
         # detection
         feats = cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000, qualityLevel=0.01,minDistance=3)
         #extraction
         kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1], _size=20) for f in feats]
         kps, des = self.orb.compute(img,kps)
         #matches
         ret = []
         if self.last is not None:
            matches = self.bf.knnMatch(des,self.last['des'], k=2)
            for m,n in matches:
             if m.distance < 0.75*n.distance:
               ret.append((kps[m.queryIdx], self.last['kps'][m.trainIdx]))

         self.last = {'kps':kps,'des':des}
         return ret
        
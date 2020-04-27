import cv2
import sdl2.ext
from display import Display 
import numpy as np


W = 1920//2
H = 1080//2

disp = Display(W,H)

class Extractor(object):
     GX = 16//2
     GY = 12//2

     def __init__(self):
        self.orb = cv2.ORB_create(100)
        self.bf = cv2.BFMatcher()
        self.last = None
     
     def extractor(self,img):
         # detection
         feats = cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8),3000, qualityLevel=0.01,minDistance=3)
         #extraction
         kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1], _size=20) for f in feats]
         kps, des = self.orb.compute(img,kps)
         #matches
         matches = None
         if self.last is not None:
           matches = self.bf.match(des,self.last['des'])
         
         self.last = {'kps':kps,'des':des}
         return kps, des, matches
        
        
                
fe = Extractor()

def process_frame(img):
    img = cv2.resize(img,(W,H))
    kps, des, matches = fe.extractor(img)
    for p in kps:
       #u, v = map(lambda x: int(round(x)), p.pt)
       u, v = map(lambda x: int(round(x)), p.pt)
       cv2.circle(img,(u,v), color=(0,255,0),radius=3)
    disp.paint(img)

    

if __name__ == "__main__":
    cap = cv2.VideoCapture("test.mp4")
    while cap.isOpened():
     ret, frame = cap.read()
     if ret == True:
        process_frame(frame)
     else:
        break
       
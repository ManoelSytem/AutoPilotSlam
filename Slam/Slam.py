import cv2
import sdl2.ext
from display import Display 
import numpy as np
from frame import Frame, denormalize, match_frames,IRt
import g2o


#camera intrinsics 
W, H = 1920//2, 1080//2

F = 270
k = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])


#main classes
disp = Display(W,H)

class Point(object):
  # A Point is 3-D point in the world
   def __init__(self, loc):
      self.frames = []
      self.location = loc
      self.idx = []

   def add_observation(self, frame, idx):
     self.frames.append(frame)
     self.idx.append(idx)

def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3],pose2[:3],pts1.T,pts2.T).T
   

frames = []
def process_frame(img):
    img = cv2.resize(img,(W,H))
    frame = Frame(img,k)   
    frames.append(frame)
    if len(frames) <= 1:
      return

    idx1, idx2, Rt = match_frames(frames[-1], frames[-2])
    frames[-1].pose  = np.dot(Rt , frames[-2].pose)
    
    #homogenous 3-D COORDS
    pts4d = triangulate(frames[-1].pose,frames[-2].pose,frames[-1].pts[idx1],frames[-2].pts[idx2])
    pts4d /= pts4d[:, 3:]
    
    #reject points behind the camera
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)
   
    for i,p in enumerate(pts4d):
      if not good_pts4d[i]:
         continue
      pt = Point(p)
      pt.add_observation(frames[-1],idx1[i]) 
      pt.add_observation(frames[-2],idx2[i])
    
    for pt1, pt2 in zip(frames[-1].pts[idx1],frames[-2].pts[idx2]):
     u1, v1 = denormalize(k,pt1)
     u2, v2 = denormalize(k,pt2)
     cv2.circle(img,(u1,v1), color=(0,255,0),radius=3)
     cv2.line(img,(u1,v1),(u2,v2), color=(255,0,0))
    disp.paint(img)
    
if __name__ == "__main__":
    cap = cv2.VideoCapture("test.mp4")
    while cap.isOpened():
     ret, frame = cap.read()
     if ret == True:
        process_frame(frame)
     else:
        break
       
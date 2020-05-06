import cv2
import sdl2.ext
from display import Display 
import numpy as np
from frame import Frame, denormalize, match
import g2o


#camera intrinsics 
W, H = 1920//2, 1080//2

F = 270
k = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])

#main classes
disp = Display(W,H)

frames = []
def process_frame(img):
    img = cv2.resize(img,(W,H))
    frame = Frame(img,k)   
    frames.append(frame)
    if len(frames) <= 1:
      return

    ret, Rt = match(frames[-1], frames[-2])

    for pt1, pt2 in ret:
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
       
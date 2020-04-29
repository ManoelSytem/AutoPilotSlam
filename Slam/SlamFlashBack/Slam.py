import cv2
import sdl2.ext
from display import Display 
from extractor import Extractor


W = 1920//2
H = 1080//2

disp = Display(W,H)
fe = Extractor()

def process_frame(img):
    img = cv2.resize(img,(W,H))
    matches = fe.extractor(img)
    if matches is None:
      return
    for pt1, pt2 in matches:
       #u, v = map(lambda x: int(round(x)), p.pt)
       u1, v1 = map(lambda x: int(round(x)), pt1.pt)
       u2, v2 = map(lambda x: int(round(x)), pt2.pt)
      
       u1 += img.shape[0]
       u2 += img.shape[0]
       v1 += img.shape[1]
       v2 += img.shape[1]
       
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
       
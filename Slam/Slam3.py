import time
import cv2
import pygame
import sdl2.ext

sdl2.ext.init()

W = 1920//2
H = 1080//2

window = sdl2.ext.Window("Slam auto pilot", size=(W, H), position=(0,500))
window.show()


def process_frame(img):
    img = cv2.resize(img,(W,H))
    events = sdl2.ext.get_events()
    for event in events:
      if event.type == sdl2.SDL_QUIT:
        exit(0)
    surf = sdl2.ext.pixels2d(window.get_surface())
    surf[:] = img.swapaxes(0,1)[:, :, 0]  
    window.refresh()        
    
    
if __name__ == "__main__":
    cap = cv2.VideoCapture('test.mp4')
    while cap.isOpened():
     ret, frame = cap.read()
     if ret == True:
        process_frame(frame)
     else:
        break
import time
import cv2
import pygame
import sdl2.ext

W = 1920//2
H = 1080//2

#pygame.init()
#display = pygame.display.set_mode((W,H))
#surface = pygame.Surface((W,H)).convert()
sdl2.ext.init()

#window = sdl2.ext.Window("Twitch Slam!",size=(W,H))
#window.show()

#cv2.namedWindow('image',cv2.WINDOW_NORMAL)

def process_frame(img):
    img = cv2.resize(img,(W,H))
    events = sdl2.ext.get_events()
    cv2.imshow('image',img)    
    print(img.shape)
    print(img)
    
    """
    #surface.blit((img.swapaxes(0,1)))
    pygame.pixelcopy.array_to_surface(surface,img.swapaxes(0,1))
    pygame.draw.circle(display,(255,0,0),(10,10),5,1)
    #pygame.surfarray.blit_array(display,img,swapaxes(0,1))
    display.blit(surface,(0,0))
    #pygame.display.flip()
    pygame.display.update()
    #time.sleep(1)
    """
    
    

if __name__ == "__main__":
    cap = cv2.VideoCapture('test.mp4')
    while cap.isOpened():
     ret, frame = cap.read()
     if ret == True:
        process_frame(frame)
     else:
        break
import cv2
import tensorflow
import numpy as np
import threading

video = cv2.VideoCapture(r'C:\github\AIprojects\firedetection\fireV2.mp4')

def t1 (threading.thread):
    ret, vid = video.read()

def t2 (threading.thread):
    
    while video.isOpened():

        cv2.namedWindow("output", cv2.WINDOW_NORMAL)

        gray = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        
        orange = np.array([255,138,0])
        lowwhite = np.array([210,210,210])
        white = np.array([255,255,255])

        mask1 = cv2.inRange(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB), orange, white)
        mask2 = cv2.inRange(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB), lowwhite, white)

        combinedmask = cv2.add(mask1, mask2)

        contours = cv2.findContours(combinedmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        x,y,w,h = cv2.boundingRect(contours[0])
        
        cv2.rectangle(combinedmask, (x,y), (x+w, y+h), (0,255,0), 3)   

        cv2.resizeWindow("output", (1400, 800)) 
        cv2.imshow("output", combinedmask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
        break
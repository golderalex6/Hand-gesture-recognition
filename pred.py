import numpy as np
from math import *
import cv2
from time import sleep 
import sklearn
import pickle
import mediapipe as mp
import time
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

label='None'
warm_up=50
label=''
i=0

with open('mlp_model.pkl','rb') as f:
    model=pickle.load(f)

mp_draw=mp.solutions.drawing_utils
mp_draw_style=mp.solutions.drawing_styles
mp_hand=mp.solutions.hands
hands=mp_hand.Hands(model_complexity=0)


def draw_bounding_box(img,hand_landmarks,w,h):
    x_max = 0
    y_max = 0
    x_min = w
    y_min = h
    lms=[]
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        lms.append([x,y])
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
    return [ (x_min, y_min), (x_max, y_max),np.array(lms,dtype=float)]

def normalize_point(box):
    x_min,y_min=box[0]
    x_max,y_max=box[1]
    points=box[2]
    points[:,0]=points[:,0]-x_min
    points[:,1]=points[:,1]-y_min
    dist_x=x_max-x_min
    dist_y=y_max-y_min
    points[:,0]=points[:,0]/dist_x
    points[:,1]=points[:,1]/dist_y
    points=points.flatten()
    points=points.tolist()
    points.append(dist_y/dist_x)
    return points

def predict_gesture(points):

    pred=model.predict([points])[0]
    label={
            0:'zero right',1:'one right',2:'two right',
            3:'three right',4:'four right',5:'five right',
            6:'oke right',7:'Hip hop right',8:'Small right',
            9:'Tym right',

            10:'zero left',11:'one left',12:'two left',
            13:'three left',14:'four left',15:'five left',
            16:'oke left',17:'Hip hop left',18:'Small left',
            19:'Tym left'


        }
    return label[pred]

prev_time=0
new_time=0

while True:
    ret, img = cap.read()
    if ret:
        img=cv2.flip(img,1)
        result=hands.process(img)
        h,w,c=img.shape
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                mp_draw.draw_landmarks(img,hand_landmarks,mp_hand.HAND_CONNECTIONS)
                box=draw_bounding_box(img,hand_landmarks,w,h)
                points=normalize_point(box)
                label=predict_gesture(points)
                cv2.putText(img,label,(box[0][0],box[0][1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),thickness=5)
        

        # cv2.putText(img,text,(30,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),thickness=5)

        new_time=time.time()
        fps=round(1/(new_time-prev_time),2)
        cv2.putText(img,f"{fps} fps",(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),thickness=5)
        prev_time=new_time

        cv2.imshow('tracking',img)

    k=cv2.waitKey(1)
    if k==ord('k'):
        break
cap.release()
cv2.destroyAllWindows()

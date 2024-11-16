import mediapipe as mp
import cv2
import time 
import numpy as np
import pandas as pd

mp_draw=mp.solutions.drawing_utils
mp_draw_style=mp.solutions.drawing_styles
mp_hand=mp.solutions.hands
hands=mp_hand.Hands()


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

warm_up=50
frame=200
i=0
label='Tym'
hand='left'
text='warm up'

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# out = cv2.VideoWriter(f'mediapipe_data/{label}_{hand}.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (100,100))

prev_time=0
new_time=0

data=[]
while True:
    ret,img=cap.read()
    if ret:
        img=cv2.flip(img,1)
        result=hands.process(img)
        h,w,c=img.shape
        if result.multi_hand_landmarks:
            if i>warm_up:
                text='filming'
            if i>frame:
                break
            i+=1
            for hand_landmarks in result.multi_hand_landmarks:

                box=draw_bounding_box(img,hand_landmarks,w,h)
                points=normalize_point(box)
                data.append(points)
                # crop_img=img[box[0][1]:box[1][1],box[0][0]:box[1][0]]
                # crop_img=cv2.resize(crop_img,(100,100))
                # out.write(crop_img)

                mp_draw.draw_landmarks(img,hand_landmarks,mp_hand.HAND_CONNECTIONS)
        

        cv2.putText(img,text,(30,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),thickness=5)

        new_time=time.time()
        fps=round(1/(new_time-prev_time),2)
        cv2.putText(img,f"{fps} fps",(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),thickness=5)
        prev_time=new_time

        cv2.imshow('tracking',img)
        if cv2.waitKey(1)==ord('k'):
            break

cap.release()
# out.release()
cv2.destroyAllWindows()
df=pd.DataFrame(data)
df.to_csv('mediapipe_data/'+label+'_'+hand+'.csv')


class hand_landmark_data():
    pass
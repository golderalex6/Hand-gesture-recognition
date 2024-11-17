import mediapipe as mp
import cv2
import time 
import numpy as np
import pandas as pd
import os
from pathlib import Path
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

class hand_landmark_data():
    def __init__(self):
        self.__mp_draw=mp.solutions.drawing_utils
        self.__mp_hand=mp.solutions.hands
        self.__hands=self.__mp_hand.Hands()

    def __draw_bounding_box(self,img,hand_landmarks,w,h):
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

    def __normalize_point(self,box):
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
    
    def record(self,label,hand,frame=200,warm_up=50):
        if not os.path.exists(os.path.join(Path(__file__).parent,'data')):
            os.mkdir(os.path.join(Path(__file__).parent,'data'))
        else:
            print("WARNING !! These is already a folder named 'data',please consider moving it to prevent data loss. Do you want to continue ?(y/n)")
            confirm=input().lower()
            if confirm!='y':
                raise Exception("These is already a folder named 'data',please consider moving it to prevent data loss.")

        cap=cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        data=[]
        for i in range(frame):
            ret,img=cap.read()
            if ret:
                img=cv2.flip(img,1)
                result=self.__hands.process(img)
                h,w,c=img.shape
                if result.multi_hand_landmarks:
                    if i>warm_up:
                        text='filming'
                        for hand_landmarks in result.multi_hand_landmarks:
                            box=self.__draw_bounding_box(img,hand_landmarks,w,h)
                            points=self.__normalize_point(box)
                            data.append(points)
                            self.__mp_draw.draw_landmarks(img,hand_landmarks,self.__mp_hand.HAND_CONNECTIONS)
                    else:
                        text='warm up'
                    cv2.putText(img,text,(30,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),thickness=5)
                cv2.imshow('tracking',img)
                if cv2.waitKey(1)==ord('k'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        df=pd.DataFrame(data)
        df.to_csv(os.path.join(Path(__file__).parent,'data',f'{label}_{hand}.csv'))

if __name__=='__main__':
    hand=hand_landmark_data()
    hand.record('heeeeeee','left')
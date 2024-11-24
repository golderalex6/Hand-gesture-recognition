import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import json
import os
from pathlib import Path

import cv2
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix 

plt.rc('figure',titleweight='bold',titlesize='large',figsize=(15,6))
plt.rc('axes',labelweight='bold',labelsize='large',titleweight='bold',titlesize='large',grid=True)

class functional():
    def __init__(self):
        pass

    def _draw_bounding_box(self,img,hand_landmarks,width,height):
        x_min,y_min,x_max,y_max=width,height,0,0
        points=[]
        for point in hand_landmarks.landmark:
            x, y = int(point.x * width), int(point.y * height)
            points.append([x,y])
            x_max=max(x_max,x)
            x_min=min(x_min,x)
            y_max=max(y_max,y)
            y_min=min(y_min,y)
        cv2.rectangle(img, (x_min, y_min),(x_max, y_max),(255,255,255),2)
        return [(x_min,y_min),(x_max,y_max),np.array(points,dtype=float)]

    def _normalize_point(self,box):
        (x_min,y_min),(x_max,y_max)=box[0:2]
        width=x_max-x_min
        height=y_max-y_min
        points=box[2]
        points[:,0]=(points[:,0]-x_min)/width
        points[:,1]=(points[:,1]-y_min)/height 
        points=np.append(points.flatten(),height/width)
        return points

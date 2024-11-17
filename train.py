import numpy as np
from math import *
import time
# from sklearn.neural_network import MLPClassifier 
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score 
import numpy as np
import pickle
import os
from pathlib import Path

# zero_r_df=pd.read_csv('mediapipe_data/Zero(fist)_right.csv',index_col=0)
# one_r_df=pd.read_csv('mediapipe_data/One_right.csv',index_col=0)
# two_r_df=pd.read_csv('mediapipe_data/Two_right.csv',index_col=0)
# three_r_df=pd.read_csv('mediapipe_data/Three_right.csv',index_col=0)
# four_r_df=pd.read_csv('mediapipe_data/Four_right.csv',index_col=0)
# five_r_df=pd.read_csv('mediapipe_data/Five_right.csv',index_col=0)
# oke_r_df=pd.read_csv('mediapipe_data/Ok_right.csv',index_col=0)
# hiphop_r_df=pd.read_csv('mediapipe_data/Hip_hop_right.csv',index_col=0)
# small_r_df=pd.read_csv('mediapipe_data/Small_right.csv',index_col=0)
# tym_r_df=pd.read_csv('mediapipe_data/Tym_right.csv',index_col=0)

# zero_l_df=pd.read_csv('mediapipe_data/Zero(fist)_left.csv',index_col=0)
# one_l_df=pd.read_csv('mediapipe_data/One_left.csv',index_col=0)
# two_l_df=pd.read_csv('mediapipe_data/Two_left.csv',index_col=0)
# three_l_df=pd.read_csv('mediapipe_data/Three_left.csv',index_col=0)
# four_l_df=pd.read_csv('mediapipe_data/Four_left.csv',index_col=0)
# five_l_df=pd.read_csv('mediapipe_data/Five_left.csv',index_col=0)
# oke_l_df=pd.read_csv('mediapipe_data/Oke_left.csv',index_col=0)
# hiphop_l_df=pd.read_csv('mediapipe_data/Hip_hop_left.csv',index_col=0)
# small_l_df=pd.read_csv('mediapipe_data/Small_left.csv',index_col=0)
# tym_l_df=pd.read_csv('mediapipe_data/Tym_left.csv',index_col=0)


# df=[
#         zero_r_df,one_r_df,two_r_df,
#         three_r_df,four_r_df,five_r_df,
#         oke_r_df,hiphop_r_df,small_r_df,tym_r_df,

#         zero_l_df,one_l_df,two_l_df,
#         three_l_df,four_l_df,five_l_df,
#         oke_l_df,hiphop_l_df,small_l_df,tym_l_df,

#     ]

# x,y=[],[]

# for i in range(len(df)):
#     dataset=df[i].values
#     n_sample=len(dataset)
#     for u in range(n_sample):
#         x.append(dataset[u])
#         y.append(i)

# x,y=np.array(x),np.array(y)
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)

# mlp=MLPClassifier(max_iter=700,activation='relu')
# mlp.fit(x_train,y_train)

# y_pred=mlp.predict(x_test)
# print(y_pred)

# acc=accuracy_score(y_test,y_pred)
# print('accuracy:',acc)

# precision=precision_score(y_test,y_pred,average='weighted')
# print('precision:',precision)

# recall=recall_score(y_test,y_pred,average='weighted')
# print('recall:',recall)

# f1=f1_score(y_test,y_pred,average='weighted')
# print('f1:',f1)

# with open('mlp_model.pkl','wb') as f:
#     pickle.dump(mlp,f)

class hand_landmark_model():
    def __init__(self):
        data_files=os.listdir(os.path.join(Path(__file__).parent,'data'))
        labels=list(map(lambda x:x.split('.')[0],data_files))

        for i in range(len(data_files)):
            df=pd.read_csv(os.path.join(Path(__file__).parent,'data',data_files[i]))
            df['label']=labels[i]
        

    def __build(self,activation='relu'):
        input_layer=tf.keras.layers.Input()
        dense_layer=tf.keras.layers.Dense(100,activation=activation)(input_layer)
        dense_layer=tf.keras.layers.Dense(50,activation=activation)(dense_layer)
        dense_layer=tf.keras.layers.Dense(20,activation=activation)(dense_layer)
        dense_layer=tf.keras.layers.Dense(10,activation)(dense_layer)
        output_layer=tf.keras.layers.Dense(10000,activation='softmax')
        
        model=tf.keras.Model(input_layer,output_layer)
        return model

    def train(self,epoch=10,batch_size=32,loss='categorical_crossentropy',optimizer='adam'):
        model=self.__build()
        model.compile(optimizer=optimizer,loss=loss)
        model.fit()
        pass
if __name__=='__main__':
    data_files=os.listdir(os.path.join(Path(__file__).parent,'data'))
    labels=list(map(lambda x:x.split('.')[0],data_files))

    df=pd.DataFrame()
    for i in range(len(data_files)):
        data=pd.read_csv(os.path.join(Path(__file__).parent,'data',data_files[i]))
        data['label']=labels[i]
    
    print(df)
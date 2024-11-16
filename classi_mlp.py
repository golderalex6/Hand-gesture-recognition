import numpy as np
from math import *
import time
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score 
import numpy as np
import pickle

zero_r_df=pd.read_csv('mediapipe_data/Zero(fist)_right.csv',index_col=0)
one_r_df=pd.read_csv('mediapipe_data/One_right.csv',index_col=0)
two_r_df=pd.read_csv('mediapipe_data/Two_right.csv',index_col=0)
three_r_df=pd.read_csv('mediapipe_data/Three_right.csv',index_col=0)
four_r_df=pd.read_csv('mediapipe_data/Four_right.csv',index_col=0)
five_r_df=pd.read_csv('mediapipe_data/Five_right.csv',index_col=0)
oke_r_df=pd.read_csv('mediapipe_data/Ok_right.csv',index_col=0)
hiphop_r_df=pd.read_csv('mediapipe_data/Hip_hop_right.csv',index_col=0)
small_r_df=pd.read_csv('mediapipe_data/Small_right.csv',index_col=0)
tym_r_df=pd.read_csv('mediapipe_data/Tym_right.csv',index_col=0)

zero_l_df=pd.read_csv('mediapipe_data/Zero(fist)_left.csv',index_col=0)
one_l_df=pd.read_csv('mediapipe_data/One_left.csv',index_col=0)
two_l_df=pd.read_csv('mediapipe_data/Two_left.csv',index_col=0)
three_l_df=pd.read_csv('mediapipe_data/Three_left.csv',index_col=0)
four_l_df=pd.read_csv('mediapipe_data/Four_left.csv',index_col=0)
five_l_df=pd.read_csv('mediapipe_data/Five_left.csv',index_col=0)
oke_l_df=pd.read_csv('mediapipe_data/Oke_left.csv',index_col=0)
hiphop_l_df=pd.read_csv('mediapipe_data/Hip_hop_left.csv',index_col=0)
small_l_df=pd.read_csv('mediapipe_data/Small_left.csv',index_col=0)
tym_l_df=pd.read_csv('mediapipe_data/Tym_left.csv',index_col=0)


df=[
        zero_r_df,one_r_df,two_r_df,
        three_r_df,four_r_df,five_r_df,
        oke_r_df,hiphop_r_df,small_r_df,tym_r_df,

        zero_l_df,one_l_df,two_l_df,
        three_l_df,four_l_df,five_l_df,
        oke_l_df,hiphop_l_df,small_l_df,tym_l_df,

    ]

x,y=[],[]

for i in range(len(df)):
    dataset=df[i].values
    n_sample=len(dataset)
    for u in range(n_sample):
        x.append(dataset[u])
        y.append(i)

x,y=np.array(x),np.array(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)

mlp=MLPClassifier(max_iter=700,activation='relu')
mlp.fit(x_train,y_train)

y_pred=mlp.predict(x_test)
print(y_pred)

acc=accuracy_score(y_test,y_pred)
print('accuracy:',acc)

precision=precision_score(y_test,y_pred,average='weighted')
print('precision:',precision)

recall=recall_score(y_test,y_pred,average='weighted')
print('recall:',recall)

f1=f1_score(y_test,y_pred,average='weighted')
print('f1:',f1)

with open('mlp_model.pkl','wb') as f:
    pickle.dump(mlp,f)



from functional import *

class hand_landmark_model(functional):
    def __init__(self):

        with open(os.path.join(Path(__file__).parent,'encode','encode.json'),'r+') as f:
            self.__label_encode=json.load(f)
            self.__labels=sorted(list(self.__label_encode.keys()),key=lambda x:self.__label_encode[x])

        with open(os.path.join(Path(__file__).parent,'metadata','model_metadata.json'),'r+') as f:
            metadata=json.load(f)
            self.__layers=metadata['layers']
            self.__activation=metadata['activation']
            self.__epochs=metadata['epochs']
            self.__loss=metadata['loss']
            self.__optimizer=metadata['optimizer']
            self.__batch_size=metadata['batch_size']

    def _build(self):
        layers=list(map(lambda x:tf.keras.layers.Dense(x,activation=self.__activation),self.__layers))
        model=tf.keras.Sequential([
                tf.keras.layers.Input(shape=(43,)),
                *layers,
                tf.keras.layers.Dense(len(self.__labels),activation='softmax')
            ])

        return model

    def train(self):

        df=pd.DataFrame()
        for file in os.listdir(os.path.join(Path(__file__).parent,'data')):
            action=pd.read_csv(os.path.join(Path(__file__).parent,'data',file))
            df=pd.concat([df,action])
        x=df.iloc[:,:-1].values
        y=df.iloc[:,-1].values
        self.__x_train,self.__x_test,self.__y_train,self.__y_test=train_test_split(x,y,test_size=0.3)

        model=self._build()
        model.compile(optimizer=self.__optimizer,loss=self.__loss)
        best_lost=tf.keras.callbacks.ModelCheckpoint(os.path.join(Path(__file__).parent,'model','hand_gesture_model.weights.h5'),save_weights_only=True,monitor='loss',mode='min',save_best_only=True)
        model.fit(self.__x_train,self.__y_train,epochs=self.__epochs,batch_size=self.__batch_size,callbacks=[best_lost])

    def evaluate(self):

        model=self._build()
        model.load_weights(os.path.join(Path(__file__).parent,'model','hand_gesture_model.weights.h5'))

        y_pred=np.argmax(model.predict(self.__x_test),axis=1)
        accuracy=accuracy_score(self.__y_test,y_pred)
        precision=precision_score(self.__y_test,y_pred,average='weighted')
        recall=recall_score(self.__y_test,y_pred,average='weighted')
        f1=f1_score(self.__y_test,y_pred,average='weighted')
        matrix=confusion_matrix(self.__y_test,y_pred)
        matrix=np.round(matrix/np.sum(matrix,axis=1,keepdims=True),2)

        print(f'Accuracy : {round(accuracy,2)}')
        print(f'Precision : {round(precision,2)}')
        print(f'Recall : {round(recall,2)}')
        print(f'F1 : {round(f1,2)}')

        fg=plt.figure()
        ax=fg.add_subplot()
        ax.imshow(matrix,cmap='Blues')
        ax.set_yticks(range(len(self.__labels)),self.__labels)
        ax.set_xticks(range(len(self.__labels)),self.__labels)
        ax.set_ylabel('True values')
        ax.set_xlabel('Predict values')
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                if matrix[y,x]>0.5:
                    color='white'
                else:
                    color='black'
                ax.text(x,y,matrix[y,x],ha='center',va='center',color=color)
        ax.set_title('Confusion matrix')
        ax.grid(False)
        plt.show()

    def gesture_predict(self):
        
        mp_draw=mp.solutions.drawing_utils
        mp_hand=mp.solutions.hands
        hands=mp_hand.Hands(model_complexity=0)
        model=self._build()
        model.load_weights(os.path.join(Path(__file__).parent,'model','hand_gesture_model.weights.h5'))

        cap=cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while True:
            ret, img = cap.read()
            if ret:
                img=cv2.flip(img,1)
                result=hands.process(img)
                height,width,_=img.shape
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:

                        mp_draw.draw_landmarks(img,hand_landmarks,mp_hand.HAND_CONNECTIONS)
                        box=self._draw_bounding_box(img,hand_landmarks,width,height)
                        points=self._normalize_point(box)
                        label=self.__labels[np.argmax(model.predict(np.array(points).reshape(1,-1)))]
                        cv2.putText(img,label,(box[0][0],box[0][1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),thickness=5)

                cv2.imshow('tracking',img)

            if cv2.waitKey(1)==ord('k'):
                break
        cap.release()
        cv2.destroyAllWindows()
if __name__=='__main__':
    landmark_model=hand_landmark_model()
    landmark_model.train()
    landmark_model.evaluate()
    landmark_model.gesture_predict()

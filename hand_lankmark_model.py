from functional import *

class hand_landmark_model(functional):
    def __init__(self):
        with open('encode.json','r+') as f:
            self.__label_encode=json.load(f)
            self.__labels=sorted(list(self.__label_encode.keys()),key=lambda x:self.__label_encode[x])

    def _build(self,activation='relu'):

        input_layer=tf.keras.layers.Input(shape=(43,))
        dense_layer=tf.keras.layers.Dense(100,activation=activation)(input_layer)
        dense_layer=tf.keras.layers.Dense(50,activation=activation)(dense_layer)
        dense_layer=tf.keras.layers.Dense(20,activation=activation)(dense_layer)
        dense_layer=tf.keras.layers.Dense(10,activation)(dense_layer)
        output_layer=tf.keras.layers.Dense(len(self.__labels),activation='softmax')(dense_layer)
        
        model=tf.keras.Model(input_layer,output_layer)
        return model

    def train(self,epochs=10,batch_size=32,loss='sparse_categorical_crossentropy',optimizer='Adam'):

        df=pd.DataFrame()
        for file in os.listdir(os.path.join(Path(__file__).parent,'data')):
            action=pd.read_csv(os.path.join(Path(__file__).parent,'data',file))
            df=pd.concat([df,action])
        x=df.iloc[:,:-1].values
        y=df.iloc[:,-1].values
        self.__x_train,self.__x_test,self.__y_train,self.__y_test=train_test_split(x,y,test_size=0.3)

        model=self._build()
        model.compile(optimizer=optimizer,loss=loss)
        best_lost=tf.keras.callbacks.ModelCheckpoint(os.path.join(Path(__file__).parent,'hand_gesture_model.weights.h5'),save_weights_only=True,monitor='loss',mode='min',save_best_only=True)
        model.fit(self.__x_train,self.__y_train,epochs=epochs,batch_size=batch_size,callbacks=[best_lost])

    def evaluate(self):

        if not os.path.exists(os.path.join(Path(__file__).parent,'hand_gesture_model.weights.h5')):
            raise Exception('There is no pre-trained model. Please train first !!')

        model=self._build()
        model.load_weights(os.path.join(Path(__file__).parent,'hand_gesture_model.weights.h5'))

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
        if not os.path.exists(os.path.join(Path(__file__).parent,'hand_gesture_model.weights.h5')):
            raise Exception('There is no pre-trained model. Please train first !!')
        
        mp_draw=mp.solutions.drawing_utils
        mp_hand=mp.solutions.hands
        hands=mp_hand.Hands(model_complexity=0)
        model=self._build()
        model.load_weights(os.path.join(Path(__file__).parent,'hand_gesture_model.weights.h5'))

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
    # landmark_model.train()
    # landmark_model.evaluate()
    landmark_model.gesture_predict()

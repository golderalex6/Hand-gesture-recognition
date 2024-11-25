from functional import *

class hand_landmark_data(functional):
    def __init__(self):
        super().__init__()
        self.__mp_draw=mp.solutions.drawing_utils
        self.__mp_hand=mp.solutions.hands
        self.__hands=self.__mp_hand.Hands()

    def record(self,label,hand,frame=200):
        #Check if exist folder named 'data'.If yes,create a warn and exit.Else,create 'data' folder
        if not os.path.exists(os.path.join(Path(__file__).parent,'data')):
            os.mkdir(os.path.join(Path(__file__).parent,'data'))
        else:
            print("WARNING !! These is already a folder named 'data',please consider moving it to prevent data loss.")

        if not os.path.exists(os.path.join(Path(__file__).parent,'encode.json')):
            open('encode.json','a+').close()

        cap=cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        raw_data=[]
        while True:
            ret,img=cap.read()
            if ret:
                img=cv2.flip(img,1)
                result=self.__hands.process(img)
                height,width,_=img.shape
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        box=self._draw_bounding_box(img,hand_landmarks,width,height)
                        points=self._normalize_point(box)
                        raw_data.append(points)
                        self.__mp_draw.draw_landmarks(img,hand_landmarks,self.__mp_hand.HAND_CONNECTIONS)
                    cv2.putText(img,'filming',(30,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),thickness=5)
                cv2.imshow('tracking',img)
                if cv2.waitKey(1)==ord('k'):
                    break
            if len(raw_data)>frame:
                break

        cap.release()
        cv2.destroyAllWindows()
        df=pd.DataFrame(raw_data)
        index=len(os.listdir(os.path.join(Path(__file__).parent,'data')))
        df['encode']=index
        df.to_csv(os.path.join(Path(__file__).parent,'data',f"{label}_{hand}.csv"),index=False)

        with open('encode.json','r+') as f:
            try:
                label_encode=json.load(f)
            except:
                label_encode={}
            if not f"{label}_{hand}" in label_encode.keys():
                label_encode[f"{label}_{hand}"]=index
                f.seek(0)
                json.dump(label_encode,f,indent=4)

if __name__=='__main__':
    hand=hand_landmark_data()
    hand.record('hi','right')

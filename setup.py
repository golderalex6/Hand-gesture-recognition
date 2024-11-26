import os
from pathlib import Path
import json

#create folder 'data' to store data
if not os.path.exists(os.path.join(Path(__file__).parent,'data')):
    os.mkdir(os.path.join(Path(__file__).parent,'data'))

#create folder 'encode' and 'encode/encode.json' to store label encode
if not os.path.exists(os.path.join(Path(__file__).parent,'encode','encode.json')):
    if not os.path.exists(os.path.join(Path(__file__).parent,'encode')):
        os.mkdir(os.path.join(Path(__file__).parent,'encode'))
    open(os.path.join(Path(__file__).parent,'encode','encode.json'),'a+').close()

#create folder 'model' to store trainned model
if not os.path.exists(os.path.join(Path(__file__).parent,'model')):
    os.mkdir(os.path.join(Path(__file__).parent,'model'))

#create folder 'metadata' to store model hyper:q
if not os.path.exists(os.path.join(Path(__file__).parent,'metadata')):
    os.mkdir(os.path.join(Path(__file__).parent,'metadata'))

    #set default hyperparameter for model
    hyperparameters={
            'layers':[100,50,20,10],
            'activation':'relu',
            'loss':'sparse_categorical_crossentropy',
            'optimizer':'adam',
            'epochs':10,
            'batch_size':32
        }
    with open(os.path.join(Path(__file__).parent,'metadata','model_metadata.json'),'a+') as f:
        json.dump(hyperparameters,f,indent=4)

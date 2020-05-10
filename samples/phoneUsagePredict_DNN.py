
# import the necessary packages

from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from collections import deque

import numpy as np
import pickle
import cv2
import os



# ...........................................................................

                                        # basic setup
labels          = ["not using phone", "using phone"]
qsize           = 32
videoName       = '20200507_144149000_iOS.MOV'
outName         = videoName[:-4]+'_'+str(qsize)+'.avi'
modelname       = 'phone_usage_DNN'                            

modelpath       = os.path.join('model',modelname+".hdf5")
pickpath        = os.path.join('model',modelname+".pickle")
videopath       = os.path.join('example_clips',videoName)
outpath         = os.path.join('output',outName)

imgMean         = np.array([123.68,116.779,103.939], 
                           dtype="float32")
Q               = deque(maxlen=qsize)

# ...........................................................................

print("Loading model and pickle files ...")
lb              = pickle.loads(open(pickpath, "rb").read())
optmz           = SGD(lr=1e-4,
                      momentum=0.9,
                      decay=1e-4/25)

#base            = ResNet50(weights="imagenet",
#                           include_top=False,
#                           input_tensor=Input(shape=(224, 224, 3)))

base            = ResNet50(weights=None,
                           include_top=False,
                           input_tensor=Input(shape=(224, 224, 3)))

def createModel():
    h   = base.output
    h   = AveragePooling2D(pool_size=(7, 7))(h)
    h   = Flatten(name="flatten")(h)
    h   = Dense(512, activation="relu")(h)
    h   = Dropout(0.5)(h)
    h   = Dense(len(lb.classes_), activation="softmax")(h)


    model = Model(inputs=base.input, outputs=h)


    for layer in base.layers:
        layer.trainable     = False

    
    return model



                            # Setup the models
model       = createModel()
model.load_weights(modelpath)
model.compile(loss='categorical_crossentropy', 
              optimizer=optmz, 
              metrics=['accuracy'])




# .............................................................................



print("Analyzing video ...")
vs      = cv2.VideoCapture(videopath)
writer  = None
(W, H)  = (None, None)


while True:
    (grabbed,
     frame)     = vs.read()


    if not grabbed:
        break
                                # Initialization on the W and H
    if W is None or H is None:
        (H, W)  = frame.shape[:2]

                                # Get the frame
    output      = frame.copy()
    frame       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame       = cv2.resize(frame, (224, 224)).astype("float32")
    frame      -= imgMean

                                # Perform prediction on the frame
    preds       = model.predict(np.expand_dims(frame,axis=0))[0]
    Q.append(preds)
                                # The size of the output of np.array(Q) is (32,3)
    predout     = np.array(Q).mean(axis=0)
    clss        = np.argmax(predout)
    label       = lb.classes_[clss]

    text        = "Event: {}".format(label)
    cv2.putText(output,
                text,
                (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.25,
                (0,255,0),
                5,
                cv2.LINE_AA)# This must be removed if requires real-time display

                            # Initialize the writer if this is not done
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(outpath,
                                 fourcc,
                                 30,
                                 (W, H),
                                 True)

                            # Write the output frame to disk
    writer.write(output)

                            # Real-time display the output image
#    cv2.imshow("Output", output)


print("Closing ...")
writer.release()
vs.release()
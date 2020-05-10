

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger



from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
import sklearn.metrics as metrics



# ...........................................................................

                                        # basic setup
labels          = ["not using phone", "using phone"]
fdr             = '../data'    # The folder that holds all the data

imgMean         = np.array([123.68,116.779,103.939], 
                           dtype="float32")

                            # Set up 'ggplot' style
plt.style.use('ggplot')     # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family']     = 'Arial'



# ...........................................................................


                            # Load the related images and resize the
                            # images into (224,224,3)
print("Preprocessing images...")
paths.image_types = (".MOV", ".mp4", ".mov")
imgPaths        = list(paths.list_images(fdr))
dat             = []
lbl             = []


for pth in imgPaths:
                                	    # Extract the label from the path
    l           = pth.split(os.path.sep)[-2]
    
    video = cv2.VideoCapture(pth)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for pthFrame in range(length):
        ok, img     = video.read()
        img         = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img         = cv2.resize(img, (224, 224))
    
        dat.append(img)
        lbl.append(l)

# ...........................................................................

                                        # Split the data into trainig and testing set
                                        # Create data generator for training and testing
    
print("Preparing data ...")
                                        # Convert the data and labels into
                                        # numpy array
dat             = np.array(dat)
lbl             = np.array(lbl)

lb              = LabelBinarizer()
lbl             = lb.fit_transform(lbl)
lbl             = to_categorical(lbl)

(trDat, 
 tsDat, 
 trLbl, 
 tsLbl)         = train_test_split(dat,
                                   lbl,
                                   test_size=0.25, 
                                   stratify=lbl,
                                   random_state=331)

trDatGen        = ImageDataGenerator(rotation_range=30,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     zoom_range=0.15,                                     
                                     shear_range=0.15,
                                     horizontal_flip=True,
                                     fill_mode="nearest")

tsDatGen        = ImageDataGenerator()

trDatGen.mean   = imgMean
tsDatGen.mean   = imgMean


# ...........................................................................


optmz           = SGD(lr=1e-4,
                      momentum=0.9,
                      decay=1e-4/25)

base            = ResNet50(weights="imagenet",
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


    model.compile(loss="categorical_crossentropy", 
                  optimizer=optmz,
                  metrics=["accuracy"])
    
    return model



                            # Setup the models
model       = createModel() # This is meant for training
modelGo     = createModel() # This is used for final testing

model.summary()


# .............................................................................


                            # Create checkpoint for the training
                            # This checkpoint performs model saving when
                            # an epoch gives highest testing accuracy
modelname       = 'phone_usage_DNN'                            
modelpath       = os.path.join('model',modelname+".hdf5")
checkpoint      = ModelCheckpoint(modelpath, 
                                  monitor='val_loss', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='min')

                            # Log the epoch detail into csv
csv_logger      = CSVLogger(modelname +'.csv')
callbacks_list  = [checkpoint,csv_logger]



pickpath        = os.path.join('model',modelname+".pickle")
f               = open(pickpath,"wb")
f.write(pickle.dumps(lb))
f.close()

# ..........................................................................

print("Start of training ...")
model.fit_generator(trDatGen.flow(trDat, trLbl, batch_size=32),
                    steps_per_epoch=len(trDat)//32,
                    validation_data=tsDatGen.flow(tsDat, tsLbl),
                    validation_steps=len(tsLbl)//32,
                    epochs=30,
                    callbacks=callbacks_list)

# ......................................................................


                            # Now the training is complete, we get
                            # another object to load the weights
                            # compile it, so that we can do 
                            # final evaluation on it
modelGo.load_weights(modelpath)
modelGo.compile(loss='categorical_crossentropy', 
                optimizer=optmz, 
                metrics=['accuracy'])

# .......................................................................

                            # Make classification on the test dataset


print("Evaluating network...")
predictions     = modelGo.predict(tsDat, batch_size=32)
predout         = np.argmax(predictions,axis=1)
testout         = np.argmax(tsLbl,axis=1)

testScores      = metrics.accuracy_score(testout,predout)
confusion       = metrics.confusion_matrix(testout,predout)

print("Best accuracy (on testing dataset): %.2f%%" % (testScores*100))
print(metrics.classification_report(testout,predout,target_names=lb.classes_,digits=4))
print(confusion)

# ..........................................................................

import pandas as pd

records     = pd.read_csv(modelname +'.csv')
plt.figure()
plt.subplot(211)
plt.plot(records['val_loss'])
plt.plot(records['loss'])
plt.yticks([0,0.20,0.40,0.60,0.80,1.00])
plt.title('Loss value',fontsize=12)

ax          = plt.gca()
ax.set_xticklabels([])

plt.subplot(212)
plt.plot(records['val_acc'])
plt.plot(records['acc'])
plt.yticks([0.6,0.7,0.8,0.9,1.0])
plt.title('Accuracy',fontsize=12)
plt.show()

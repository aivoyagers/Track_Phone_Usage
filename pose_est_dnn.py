# -*- coding: utf-8 -*-
"""
Created on Tue May  5 19:24:54 2020

@author: rajas
"""
import pandas as pd
import numpy as np
import json
import os
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.svm import SVC
import joblib

import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

warnings.filterwarnings("ignore")

# ...........................................................................

data_dir = "output"
modelname = "pose_track_dnn"

                            # Set up 'ggplot' style
plt.style.use('ggplot')     # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family']     = 'Arial'

links = [[(18, 19, 20), (24, 25, 26), (30, 31, 32)],  # Right arm
         [(15, 16, 17), (21, 22, 23), (27, 28, 29)]]  # Left arm
    
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    ang = ang + 360 if ang < 0 else ang
    ang = 360 - ang if ang > 180 else ang 
    return ang

def getListOfArmAngles(ptLists):
    rArmAngle = []
    lArmAngle = []
    for ptList in ptLists:
        # measure right arm angle and left arm angle
        for i, (a0, a1, a2) in enumerate(links) :
            pt0     = np.int32([ptList[a0[0]], ptList[a0[1]]])
            pt1     = np.int32([ptList[a1[0]], ptList[a1[1]]])
            pt2     = np.int32([ptList[a2[0]], ptList[a2[1]]])                
            
            if i == 0:
                rArmAngle.append(getAngle(pt0, pt1, pt2))
            else:
                lArmAngle.append(getAngle(pt0, pt1, pt2))                
    return (rArmAngle, lArmAngle)


# with open(data_file, 'r') as f:
#     data = json.load(f)
# df = pd.DataFrame(data)
data = []
print('Loading pose data..')
for file_name in [file for file in os.listdir(data_dir) if file.endswith('.json')]:
  with open(os.path.join(data_dir, file_name), 'r') as json_file:
    data.extend(json.load(json_file))

df = pd.DataFrame(data)    
print('Loading pose data completed..')

y       = df['category_id']
le      = LabelEncoder()
y       = le.fit_transform(y)
classes = list(le.classes_)                 # the output is a funny numpy str_ object
classes = [str(c) for c in classes]         # convert each output in the list to string
y       = to_categorical(y,num_classes=len(classes)) 

# ............................................................................

print('Preparing data for training..')
X = pd.DataFrame(df['keypoints'].tolist())
X['score'] = df['score']

rArmAngle, lArmAngle = getListOfArmAngles(df['keypoints'])
X['rArmAngle']       = rArmAngle
X['lArmAngle']       = lArmAngle
inputLength          = len(X.columns) 

trDat, vlDat, trLbl, vlLbl = train_test_split(X, y, test_size = 0.20, random_state = 50, stratify=y, shuffle=True)

# ...........................................................................

                                            # Create the deep learning model
def createModel(inputSize):
    ipt = Input(shape=(inputSize,1))

    x   = Conv1D(8, 11, padding='valid', activation='relu')(ipt)
    x   = MaxPooling1D(4)(x)
    x   = Dropout(0.25)(x)
    
    x   = Conv1D(16, 11, padding='valid', activation='relu')(x)
    x   = MaxPooling1D(4)(x)
    x   = Dropout(0.25)(x)
    
    x   = Conv1D(32, 11, padding='valid', activation='relu')(x)
    x   = MaxPooling1D(4)(x)
    x   = Dropout(0.25)(x)
    
    x   = Conv1D(64, 11, padding='valid', activation='relu')(x)
    x   = MaxPooling1D(4)(x)
    x   = Dropout(0.25)(x)
    
    x   = Flatten()(x)
    x   = Dense(256, activation='relu')(x)
    x   = Dropout(0.5)(x)
    
    x   = Dense(128, activation='relu')(x)
    x   = Dropout(0.5)(x)
    
    x   = Dense(len(classes), activation='softmax')(x)
    
    model = Model(ipt, x)
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    return model

                                        # Setup the models
model       = createModel(inputLength) # This is meant for training
modelGo     = createModel(inputLength) # This is used for final testing

model.summary()

plot_model(model, 
           to_file=modelname+'_plot.pdf', 
           show_shapes=True, 
           show_layer_names=False,
           rankdir='TB')

# .............................................................................
 
                            # Create checkpoint for the training
                            # This checkpoint performs model saving when
                            # an epoch gives highest testing accuracy
filepath        = modelname + ".hdf5"
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor='val_loss', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='min')

                            # Log the epoch detail into csv
csv_logger      = CSVLogger(modelname +'.csv')
callbacks_list  = [checkpoint,csv_logger]

# .............................................................................

                            # Fit the model
                            # This is where the training starts
model.fit(trDat, 
          trLbl, 
          validation_data=(vlDat, vlLbl), 
          epochs=100, 
          batch_size=32,
          shuffle=True,
          callbacks=callbacks_list)

# ......................................................................
                            # Now the training is complete, we get
                            # another object to load the weights
                            # compile it, so that we can do 
                            # final evaluation on it
modelGo.load_weights(filepath)
modelGo.compile(loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])

# .......................................................................



                            # Make classification on the test dataset
predicts    = modelGo.predict(vlDat)

                            # Prepare the classification output
                            # for the classification report
predout     = np.argmax(predicts,axis=1)
testout     = np.argmax(vlLbl,axis=1)

testScores  = metrics.accuracy_score(testout,predout)
confusion   = metrics.confusion_matrix(testout,predout)

print("Best accuracy (on testing dataset): %.2f%%" % (testScores*100))
print(metrics.classification_report(testout,predout,target_names=classes,digits=4))
print(confusion)

#------------------------------------------------------------------------------

records     = pd.read_csv(modelname +'.csv')
plt.figure()
plt.subplot(211)
plt.plot(records['val_loss'])
plt.plot(records['loss'])
plt.yticks([0.00,0.40,0.60,0.80])
plt.title('Loss value',fontsize=12)

ax          = plt.gca()
ax.set_xticklabels([])



plt.subplot(212)
plt.plot(records['val_acc'])
plt.plot(records['acc'])
plt.yticks([0.6,0.7,0.8,0.9])
plt.title('Accuracy',fontsize=12)
plt.show()

# ............................................................................
'''
print('Training started..')
svc_model = SVC(kernel = 'linear', C = 10).fit(X_train, y_train)

# Save the model for run-time use
joblib.dump(svc_model, model_file)

# Load from file
loaded_svc_model = joblib.load(model_file)

y_pred = loaded_svc_model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print ("Accuracy Score : ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
'''

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
#from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
import joblib

def pose_track_train() :
    data_dir = "pose_points_data"
    model_file = os.path.join(data_dir, "phone_view_detection_svc_1-0.joblib")
    
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
    
    y = df['category_id']
    # lbl = preprocessing.LabelEncoder()
    # lbl.fit(y)
    # lbl.classes_
    # y = lbl.transform(y)
    
    print('Preparing data for training..')
    X = pd.DataFrame(df['keypoints'].tolist())
    X['score'] = df['score']
    rArmAngle, lArmAngle = getListOfArmAngles(df['keypoints'])
    X['rArmAngle'] = rArmAngle
    X['lArmAngle'] = lArmAngle
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 50, stratify=y)
    
    print('Training started..')
    svc_model = SVC(kernel = 'linear', C = 10).fit(X_train, y_train)
    print('Model training completed..')
    # Save the model for run-time use
    joblib.dump(svc_model, model_file)
    
    # Load from file
    loaded_svc_model = joblib.load(model_file)
    
    y_pred = loaded_svc_model.predict(X_test)
    
    print("==============================================================================\n")
    
    print(confusion_matrix(y_test, y_pred))
    print ("Accuracy Score : ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    print(f'weighted f1 score         : {f1_score(y_test, y_pred, average="macro"):.4f}') 
    print(f'weighted precision score  : {precision_score(y_test, y_pred, average="macro"):.4f}')
    print(f'weighted recall score     : {recall_score(y_test, y_pred, average="macro"):.4f}') 
    print("\n==============================================================================")    
        
    
    
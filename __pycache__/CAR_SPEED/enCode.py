import pickle
import cv2
import face_recognition
import os
import numpy as np
import time
import base64

folderPath = 'img'
pathList = os.listdir(folderPath)

imgList =[]
studentIds= []

for path in pathList:
    encodeList = []
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    studentIds.append(os.path.splitext(path)[0])

def findEncodingimgs(imgList):
    for img in imgList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    
    return encodeList
encodeLists = findEncodingimgs(imgList)
encodeListIds = [encodeList,studentIds]
print(encodeLists)
print(studentIds)

# file = open("EncodeFile.p",'wb')
# pickle.dump(encodeListIds,file)
# file.close()
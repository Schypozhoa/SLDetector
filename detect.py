from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import math

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 10)
detector = HandDetector(detectionCon=0.8, maxHands=1)
model = Classifier("F:/UwU/!DATA/SIB - AI Mastery/TA/ta-env/File/Model/mobilenet_datasetASL.h5")
abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
offset = 20
imgSize = 224

while True:
    key=cv2.waitKey(1)
    success, img = cap.read(1)
    hands, img = detector.findHands(img)
    if hands:
        hand1 = hands[0]
        x, y, w, h = hand1["bbox"]
        lmList1 = hand1["lmList"]
        
        imgCrop1 = img[y-offset:y+h+offset, x-offset:x+w+offset]
        imgWhite1 = np.ones((imgSize, imgSize, 3), np.uint8)*255
        
        if h/w > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize1 = cv2.resize(imgCrop1, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite1[:, wGap:wCal+wGap] = imgResize1
            pred, index = model.getPrediction(imgWhite1, draw=False)
            
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize1 = cv2.resize(imgCrop1, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite1[hGap:hCal+hGap, :] = imgResize1
            pred, index = model.getPrediction(imgWhite1, draw=False)
            print(pred, index, abc[index])
        
        image = cv2.putText(imgWhite1, f'{abc[index]} {pred[index]*100:.2f}%', (0, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,189,86), 2)
        cv2.imshow("Hand Crop Screcth", imgWhite1)
    
    cv2.namedWindow("Prediction", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Prediction", img)
    
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

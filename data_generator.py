from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import math

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 10)
detector = HandDetector(detectionCon=0.8, maxHands=1)
offset = 20
imgSize = 224
train_path = "ASL Hand Language/"
abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
idx = 17
iter = 0
saving = False

while True:
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
            
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize1 = cv2.resize(imgCrop1, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite1[hGap:hCal+hGap, :] = imgResize1
        
        cv2.imshow("Hand Crop Screcth", imgWhite1)
    
    if saving:
        cv2.imwrite(f"{train_path}{abc[idx]}/{abc[idx]}{iter}.jpg", imgWhite1)
        print(f"Saving {abc[idx]}{iter}...")
        iter+=1
        if iter%50 == 0:
            print("Change the background pls...")
            saving = False
        if iter == 150:
            idx+=1
            iter=0
            saving = False
            print(f"Changing to {abc[idx]}, iter reset currently at {iter}")
    cv2.namedWindow("Prediction", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Prediction", img)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('s'):
        if saving:
            saving = False
        else:
            saving = True
    if key == ord('i'):
        print("Currently in letter\t-> ", abc[idx])
        print("Currently in iter\t-> ", iter)
    
cap.release()
cv2.destroyAllWindows()

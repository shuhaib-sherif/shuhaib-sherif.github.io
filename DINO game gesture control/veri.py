import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os
from keras.models import load_model
import pyautogui

# Loading the model
model=load_model("model_hands1.h5")
cap = cv2.VideoCapture(0)

# Category dictionary
categories = {0: 'ZERO',  1: 'FIVE'}

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]

    # Resizing the ROI so it can be fed to the model for prediction5
    roi = cv2.resize(roi, (224, 224))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    from keras.preprocessing import image
    x = image.img_to_array(test_image)
    x = np.expand_dims(x, axis=0)
    # print(x.shape)

    # Batch of 1



    # print(x.shape)
    result = model.predict(x)



    prediction = {'ZERO': result[0][0],
                  # 'ONE': result[0][1],
                  # 'TWO': result[0][2],
                  # 'THREE': result[0][3],
                  # 'FOUR': result[0][4],
                  'FIVE': result[0][1]}
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 5, (0,255,255), 3)
    cv2.imshow("Frame", frame)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break


cap.release()
cv2.destroyAllWindows()

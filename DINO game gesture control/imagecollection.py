import cv2
import numpy as np
import os

# Create the directory structure
if not os.path.exists("Datasets"):
    os.makedirs("Datasets")
    os.makedirs("Datasets/Train")
    os.makedirs("Datasets/Test")
    os.makedirs("Datasets/Train/0")
    os.makedirs("Datasets/Test/0")
    os.makedirs("Datasets/Train/5")
    os.makedirs("Datasets/Test/5")


# Train or test
mode_train = 'Train'
directory_train = 'Datasets/'+mode_train+'/'

mode_test = 'Test'
directory_test = 'Datasets/'+mode_test+'/'


cap = cv2.VideoCapture(0)

count_0 = 0
count_5 = 0

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # Getting count of existing images
    count__train = {'zero_train': len(os.listdir(directory_train+"/0")),
                    'five_train': len(os.listdir(directory_train+"/5"))}

    count__test = {'zero_test': len(os.listdir(directory_test+"/0")),
                   'five_test': len(os.listdir(directory_test+"/5"))}

    # Printing the count in each set to the screen
    cv2.putText(frame, "IMAGE COUNT", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ZERO TRAIN : "+str(count__train['zero_train']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ZERO TEST: "+str(count__test['zero_test']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "FIVE TRAIN: "+str(count__train['five_train']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "FIVE TEST: "+str(count__test['five_test']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

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
    roi = cv2.resize(roi, (224, 224))

    cv2.imshow("Frame", frame)

    # do the processing after capturing the image!
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", roi)



    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break

    if interrupt & 0xFF == ord('0'):

        if count_0<=500:

            cv2.imwrite(directory_train+'0/'+str(count__train['zero_train'])+'.jpg', roi)


        elif count_0>500 and count__test['zero_test']<=100:

            cv2.imwrite(directory_test+'0/'+str(count__test['zero_test'])+'.jpg', roi)


        count_0 = count_0 + 1
        # print(count)

    if interrupt & 0xFF == ord('5'):


        if count_5<=500:

            cv2.imwrite(directory_train+'5/'+str(count__train['five_train'])+'.jpg', roi)


        elif count_5>500 and count__test['five_test']<=100:

            cv2.imwrite(directory_test+'5/'+str(count__test['five_test'])+'.jpg', roi)

        count_5 = count_5 + 1

cap.release()
cv2.destroyAllWindows()

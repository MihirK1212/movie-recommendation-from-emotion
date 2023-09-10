import tensorflow as tf
import cv2
import os
import numpy as np
from tensorflow import keras
import cv2
import numpy as np

Classes=["Happy","Negative","Shocked"]

path='haarcascade_frontalface_default.xml'
model=keras.models.load_model('emotion_detector.h5')

size = 4
webcam = cv2.VideoCapture(0) #Use camera 0

# We load the xml file
classifier = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')

while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,1) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        #Save just the rectangle faces in SubRecFaces
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(96,96))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,96,96,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        
        label=np.argmax(result,axis=1)[0]

        prob=str((result[0][label])*100)

        prob_val=((result[0][label])*100)
        # if prob_val>90:
        #     print(prob_val)



        class_label=Classes[label]

        cv2.rectangle(im,(x,y-40),(x+w,y),(0,0,0),-1)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,0),2)
        cv2.putText(im,class_label, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)


    # Show the image
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop
    if key == 27: #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()

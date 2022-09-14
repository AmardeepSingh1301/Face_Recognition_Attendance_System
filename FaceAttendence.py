import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images'
images = []
personName = []
myList = os.listdir(path)
print(myList)
for cu_img in myList:
    current_img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_img)
    personName.append(os.path.splitext(cu_img)[0])

#print(personName)

def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = faceEncodings(images)
print(" All Encoding Completed!!!!!")

def attendence(name):
    already_in_file = set()
    with open('attendence.csv', "r") as g:  # just read
        for line in g:
            already_in_file.add(line.split(",")[0])

    # process your current entry:
    if name not in already_in_file:
        with open('attendence.csv', "a") as g:  # append
            now = datetime.now()
            timeString = now.strftime('%H:%M:%S')
            dateString = now.strftime('%d-%m-%Y')
            g.writelines(f'\n{name},{timeString},{dateString}')

# Capture image from camera
cam = cv2.VideoCapture(1)

while True:
    ret, frame = cam.read(1)
    faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)
        print(matchIndex)

        if matches[matchIndex]:
            name = personName[matchIndex].upper()
            #print(name)

            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0, 255, 0), 1)
            cv2.rectangle(frame, (x1, y2-35), (x2, y2),(0,255,0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            attendence(name)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == 13:
        break

cam.release()
cv2.destroyAllWindows()


















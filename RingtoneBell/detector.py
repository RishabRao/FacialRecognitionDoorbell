import cv2
import numpy as np
import sqlite3
import os


def main():
    pass


def detect_faces():
    database = sqlite3.connect('database.db')
    dataSearcher = database.cursor()

    training_file = "recognizer/trainingData.yml"

    if not os.path.isfile(training_file):
        print("Please train the file!")
        exit(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(training_file)
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            ids, conf = recognizer.predict(gray[y:y + h, x:x + w])
            dataSearcher.execute("select name from users where id = (?);", (ids,))
            result = dataSearcher.fetchall()
            name = result[0][0]
            if conf < 50:
                cv2.putText(img, name, (x + 2, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 0), 2)

                cv2.putText(img, 'Welcome ' + name + ' Ring the doorbell by pressing the button!', (200, 660),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, 'No Match', (x + 2, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Face Recognizer', img)
        k = cv2.waitKey(33)
        if k == 27:  # Esc key to stop
            break

    cv2.destroyAllWindows()
    return

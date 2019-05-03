import cv2
import numpy as np
import sqlite3
import os


def main():
    pass


def record_faces():
    database = sqlite3.connect('database.db')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')

    cursor = database.cursor()

    cap = cv2.VideoCapture(0)

    name = input("What is your name?: ")
    cursor.execute('INSERT INTO users (name) VALUES (?)', (name,))
    user_id = cursor.lastrowid

    sample_image_count = 0

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sample_image_count = sample_image_count + 1
            cv2.imwrite("dataset/User." + str(user_id) + "." + str(sample_image_count) + ".jpg", gray[y:y + h, x:x + w])
            cv2.putText(img, "Recording face", (500, 660), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.waitKey(100)
        cv2.imshow('img', img)
        cv2.waitKey(1)
        if sample_image_count > 20:
            break
    cap.release()
    database.commit()
    database.close()
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return

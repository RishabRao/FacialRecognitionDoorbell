import os
import cv2  # Imported external camera vision library to help with camera processing
import numpy as np  # Imported external math library to help with arrays
from PIL import Image  # Imported Python Image Library for image processing


def main():
    """
    This is the method that would run if trainer was imported or ran individually
    :return: void
    """
    pass


def train_faces():
    """
    This method trains the faces and matches it to their names. It stores the data on the users faces in a yml file
    for use when recognition is needed.
    :return: void
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = 'dataset'
    if not os.path.exists('./recognizer'):
        os.makedirs('./recognizer')

    def getImagesWithID(path):

        imagePaths = [os.path.join(path, files) for files in os.listdir(path)]

        faces = []

        IDs = []
        for imagePath in imagePaths:
            if imagePath.endswith('jpg'):
                faceImg = Image.open(imagePath).convert('L')
                faceNp = np.array(faceImg, 'uint8')
                ID = int(os.path.split(imagePath)[-1].split('.')[1])
                faces.append(faceNp)
                IDs.append(ID)
        return np.array(IDs), faces

    Ids, faces = getImagesWithID(path)
    recognizer.train(faces, Ids)
    recognizer.save('recognizer/trainingData.yml')


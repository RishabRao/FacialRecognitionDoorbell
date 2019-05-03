import detector
import record_faces
import trainer
import create_database
import cv2
import os

if __name__ == '__main__':
    print("Welcome to the Facial Recognition doorbell!\n")

    if not os.path.exists('database.db'):
        print('We have detected this is your first time running this program! We will create a database for you!\n')
        create_database.create_database()

    running = True

    while running:
        user_input = input("\nYou can start the doorbell by typing 1\n"
                           "You can record a new face by typing 2\n"
                           "You can exit by typing 3\n"
                           "---> ")

        if user_input == "1":
            detector.detect_faces()
            cv2.destroyAllWindows()

        elif user_input == "2":
            record_faces.record_faces()
            trainer.train_faces()
            print("\nFace saved!")

        elif user_input == "3":
            running = False

        else:
            print("Invalid Input! Try again")

    print("Thank you for using the Facial Recognition doorbell!")

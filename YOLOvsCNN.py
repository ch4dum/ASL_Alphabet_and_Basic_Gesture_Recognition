import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from ultralytics import YOLO
import os

# Models and Label
MODELS = {
    'basic': {
        'cnn': 'CNN_basic.h5',
        'yolo': 'YOLO11_basic.pt',
        'alphabet': ['Hello', 'IloveYou', 'No', 'Please', 'Thanks', 'Yes']
    },
    'alphabet': {
        'cnn': 'CNN_alphabet.h5',
        'yolo': 'YOLO11_alphabet.pt',
        'alphabet': list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    }
}

def run_cnn(model_path, labels):
    model = load_model(model_path)

    def classify(image):
        image = cv2.resize(image, (64, 64))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        proba = model.predict(image, verbose=0)
        idx = np.argmax(proba)
        confidence = np.max(proba)
        return labels[idx], confidence

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Can't open the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        top, right, bottom, left = 75, 350, 300, 590
        roi = frame[top:bottom, right:left]
        roi = cv2.flip(roi, 1)

        try:
            prediction, confidence = classify(roi)
            if confidence < 0.6:
                prediction = "?"
        except:
            prediction = "?"

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("Press 'q' to quit | 'b' to go back", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            cap.release()
            cv2.destroyAllWindows()
            return  # go back

    cap.release()
    cv2.destroyAllWindows()

def run_yolo(model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Can't open the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.25, show=False, verbose=False)
        annotated_frame = results[0].plot()
        cv2.imshow("Press 'q' to quit | 'b' to go back", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

def main_menu():
    while True:
        print("\n--- Select Detection Mode ---")
        print("1. Basic Posture Detection")
        print("2. Alphabet Detection")
        print("3. Quit")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            sub_menu('basic')
        elif choice == '2':
            sub_menu('alphabet')
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")

def sub_menu(mode_key):
    while True:
        print(f"\n--- {mode_key.capitalize()} Mode ---")
        print("1. Use CNN Model")
        print("2. Use YOLO Model")
        print("3. Back")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            run_cnn(MODELS[mode_key]['cnn'], MODELS[mode_key]['alphabet'])
        elif choice == '2':
            run_yolo(MODELS[mode_key]['yolo'])
        elif choice == '3':
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main_menu()

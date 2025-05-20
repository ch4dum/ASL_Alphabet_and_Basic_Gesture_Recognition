import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from ultralytics import YOLO

# Models and label
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

# Functions for CNN
def run_cnn(model_path, labels):
    model = load_model(model_path, compile=False)

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
        messagebox.showerror("Error", "Cannot open webcam")
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

        cv2.imshow("Webcam - Press 'q' to quit, 'b' to go back", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Functions for YOLO
def run_yolo(model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.25, show=False, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("YOLO - Press 'q' to quit, 'b' to go back", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Front GUI
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Detection")
        self.root.geometry("300x300")
        self.mode = None

        self.main_menu()

    def clear_frame(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def main_menu(self):
        self.clear_frame()
        tk.Label(self.root, text="Select Detection Mode", font=('Helvetica', 14)).pack(pady=20)
        tk.Button(self.root, text="Basic Posture", width=20, command=lambda: self.model_menu('basic')).pack(pady=10)
        tk.Button(self.root, text="Alphabet", width=20, command=lambda: self.model_menu('alphabet')).pack(pady=10)
        tk.Button(self.root, text="Quit", width=20, command=self.root.quit).pack(pady=30)

    def model_menu(self, mode_key):
        self.clear_frame()
        self.mode = mode_key
        tk.Label(self.root, text=f"{mode_key.capitalize()} - Choose Model", font=('Helvetica', 14)).pack(pady=20)
        tk.Button(self.root, text="Use CNN", width=20,
                  command=lambda: run_cnn(MODELS[self.mode]['cnn'], MODELS[self.mode]['alphabet'])).pack(pady=10)
        tk.Button(self.root, text="Use YOLO", width=20,
                  command=lambda: run_yolo(MODELS[self.mode]['yolo'])).pack(pady=10)
        tk.Button(self.root, text="Back", width=20, command=self.main_menu).pack(pady=30)

# Start!
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

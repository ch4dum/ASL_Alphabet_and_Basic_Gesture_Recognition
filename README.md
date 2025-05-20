# ASL_Alphabet_and_Basic_Gesture_Recognition

This repository is created for submission in the course **FRA626 Machine Vision for Smart Factory**.

> A real-time sign language recognition system using **YOLOv11** and **CNN**,  
> supporting both **Alphabet (A–Z)** and **Basic Signs** like *Hello*, *Yes*, *No*, etc.

---

## Overview

The system supports two recognition modes:

- **Alphabet Mode:** A–Z (excluding J and Z)
- **Basic Sign Mode:** Hello, Thank you, Yes, No, I Love You

There are two model types:
- `YOLOv11` — object detection + classification
- `CNN` — classification from cropped hand images

You can test and compare both models via terminal or GUI.

---

## Project Structure

### Datasets
- **YOLO Format:**
  - `alphabet_dataset/`
  - `basic_dataset/`
- **CNN Format (cropped from YOLO):**
  - `cnn_alphabet_dataset/`
  - `cnn_basic_dataset/`

### Data Preparation
- `cropnclass.py`  
  Script to convert YOLO-annotated images into cropped datasets suitable for CNN training.

### Model Training
- **CNN:**
  - `CNN_alphabet-train.ipynb` – Training notebook for alphabet signs
  - `CNN_basic-train.ipynb` – Training notebook for basic signs
  - `CNN_alphabet-real_time.ipynb` - Real-Time Inference for alphabet signs
  - `CNN_basic-real_time.ipynb` - Real-Time Inference for basic signs
- **YOLOv11:**
  - `YOLO11_alphabet.ipynb` – Full pipeline for YOLOv11 (alphabet)
  - `YOLO11_basic.ipynb` – Full pipeline for YOLOv11 (basic)

### Model Comparison
- `YOLOvsCNN.py` – Terminal-based UI to test both models
- `YOLOvsCNN_tkinter.py` – Tkinter GUI for ease of use (with visual feedback)

### Trained Models
- `CNN_alphabet.h5`
- `CNN_basic.h5`
- `YOLO11_alphabet.pt`
- `YOLO11_basic.pt`

---

## How to Use
### Option 1: Run via **Terminal UI**

To run the model comparison through a **menu-based terminal interface**, use:

```bash
python YOLOvsCNN.py
````

![Image](https://github.com/user-attachments/assets/1d936ed5-264d-4b35-9e91-3f9cef2ec8d5)


* You'll be prompted to select the model type (YOLO or CNN) and the recognition mode (Alphabet or Basic).
* The selected model will activate your webcam and perform real-time recognition.

### Option 2: Run via **Tkinter GUI**

To launch a **graphical interface** with buttons and image preview, use:

```bash
python YOLOvsCNN_tkinter.py
```
![Image](https://github.com/user-attachments/assets/97612a1a-14f5-4760-bd00-82c569bec8d2)
![Image](https://github.com/user-attachments/assets/833aea45-1fff-4cb4-bc13-23d72c01c86d)

* Select between YOLO or CNN using a graphical menu.
* You can also choose Alphabet or Basic mode with a single click.
* Detection results will be shown directly in the GUI window with bounding boxes or predicted labels.

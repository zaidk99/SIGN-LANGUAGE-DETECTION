# 🖐️ Sign Language Detection using MediaPipe & LSTM

A real-time sign language detection project that uses a webcam to track hand and body movements with MediaPipe Holistic, and an LSTM model to recognize sign language gestures from motion sequences.
The system supports data collection, training, evaluation, and real-time prediction using a webcam.

## Features

- Real-time sign language detection using a webcam
- MediaPipe Holistic for:
  - Pose landmarks
  - Hand landmarks
  - Face landmarks
- LSTM-based deep learning model for sequence classification
- Modular, clean, and extensible project structure
- Easy to add new signs and retrain the model
- TensorBoard logging support

## Requirements

## Software

- Python 3.7 or higher

```bash 
pip install numpy opencv-python mediapipe tensorflow scikit-learn
```

## Or use the recommended requirements.txt:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

## Project Structure

- **signdetection/**
  - **src/**
    - `config.py` – Central configuration  
    - `mediapipe_utils.py` – MediaPipe detection helpers  
    - `data_utils.py` – Data loading and preprocessing  
    - `model_utils.py` – LSTM model   
    - `visualization.py` – Drawing and visualization  

  - **scripts/** 
    - `collect_data.py` – Dataset collection  
    - `train_model.py` – Model training  
    - `evaluate_model.py` – Model evaluation  
    - `realtime_detection.py` – Real-time sign detection  

  - **MP_Data/** – Collected dataset (auto-created)
  - `README.md`
  - `.gitignore`
  - `requirements.txt`


## ⚙️ Configuration

## Signs to detect
- signs = np.array(["please", "yes"]) - `Change the Signs Data folders name as you collect`

## Data collection
- `no_sequences = 30` - Videos per sign
- `sequence_length = 30` - Frames per video

## Training
- EPOCHS = 800
- TEST_SIZE = 0.15
- MODEL_NAME = "sign1.h5"

## Detection
- threshold = 0.8 - `Prediction confidence threshold`
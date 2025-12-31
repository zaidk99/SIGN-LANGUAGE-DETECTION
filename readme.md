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
- MODEL_NAME = "sign1.h5" - `Edit The Model Name`

## Detection
- threshold = 0.8 - `Prediction confidence threshold`


## 🚀 Quick Start

- Step 1: Collect Training Data

- Collect gesture data using your webcam:

```bash 
python scripts/collect_data.py
```

- Opens webcam
- Collects gesture sequences for each sign
- **Each sign:** - 
      - 30 sequences
	  - 30 frames per sequence
	  - Saves landmark data as .npy files inside MP_Data/


## Step 2: Train the Model

- Train the LSTM model on collected data:

```bash 
python scripts/train_model.py
```
- Loads data from MP_Data/
- Splits data into training and testing sets (85% / 15%)
- Builds and compiles the LSTM model
- Trains for configured epochs
- Saves model as sign1.h5
- Creates TensorBoard logs in detection/Logs/


## Step 3: Evaluate the Model (Optional)

- Evaluate model performance:

``` bash 
python scripts/evaluate_model.py
```
- **Outputs:** - 
    - Accuracy Score
    - Prediction vs actual comparison


## Step 4: Real-Time Detection

- Run live sign language detection:

```bash 
python scripts/realtime_detection.py
```
- Opens webcam feed
- Detects landmarks in real time
- Predicts sign every 30 frames
- **Displays:** - 
    - Probability bars
    - Detected sentence
    - Press q to quit

## 📊 Performance Tips
- Use diverse lighting conditions
- Capture multiple angles
- Minimum 30 sequences per sign
- Monitor training using TensorBoard
- Adjust confidence threshold for better predictions
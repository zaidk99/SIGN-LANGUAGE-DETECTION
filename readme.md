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

signdetection/
│
├── src/                        
│   ├── config.py...............# Central configuration
│   ├── mediapipe_utils.py      # MediaPipe detection helpers
│   ├── data_utils.py           # Data loading & preprocessing
│   ├── model_utils.py          # LSTM model
│   └── visualization.py        # Drawing & visualization helpers
│
├── scripts/                    
│   ├── collect_data.py         # Dataset collection
│   ├── train_model.py          # Model training
│   ├── evaluate_model.py       # Model evaluation
│   └── realtime_detection.py   # Real-time sign detection
│
├── MP_Data/                    # Collected dataset (auto-created)
│
├── README.md
├── .gitignore
└── requirements.txt

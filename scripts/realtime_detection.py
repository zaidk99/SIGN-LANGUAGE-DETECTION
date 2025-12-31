import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Import from config
from src.config import (
    signs,
    sequence_length,
    threshold,
    MODEL_NAME,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE
)

# Import MediaPipe functions
from src.mediapipe_utils import (
    mediapipe_detection,
    draw_styled_landmarks,
    extractk
)

# Import visualization
from src.visualization import prob_viz

# Import MediaPipe classes
mp_holistic = mp.solutions.holistic

# real-time detection
if __name__ == "__main__":
    # Load model
    model = load_model(MODEL_NAME)
    
    # Detection variables
    sequence = []
    sentence = []
    colors = [(245,117,16), (117,245,16), (16,117,245)]
    
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                              min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # Prediction logic
            keypoints = extractk(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(signs[np.argmax(res)])
                
                
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if signs[np.argmax(res)] != sentence[-1]:
                            sentence.append(signs[np.argmax(res)])
                    else:
                        sentence.append(signs[np.argmax(res)])
                
                if len(sentence) > 5:
                    sentence = sentence[-5:]
                
                
                image = prob_viz(res, signs, image, colors)
            
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
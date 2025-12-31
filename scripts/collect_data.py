import cv2
import numpy as np
import os
import mediapipe as mp

# Import from config
from src.config import (
    DATA_PATH,
    signs,
    no_sequences,
    sequence_length,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE
)

# Import MediaPipe functions
from src.mediapipe_utils import (
    mediapipe_detection,
    draw_styled_landmarks,
    extractk
)

# Import MediaPipe classes
mp_holistic = mp.solutions.holistic

#data collection 
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=MIN_DETECTION_CONFIDENCE, 
                              min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as holistic:
        for action in signs:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120,200),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(100)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed for Sign Data', image)
                    keypoints = extractk(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                        
    cap.release()
    cv2.destroyAllWindows()
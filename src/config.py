import os
import numpy as np
# Data paths
DATA_PATH = os.path.join('MP_Data')
MODEL_NAME = 'sign1.h5'
LOG_DIR = 'full_file_path_to_create_dir/detection/Logs'

# Signs to detect
signs = np.array(["please", "yes"])  #Sample Data only to view of plese and yes is given prepare your sign data and then move forward 

# Data collection parameters
no_sequences = 30
sequence_length = 30

# Model training parameters
EPOCHS = 800
TEST_SIZE = 0.15

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Real-time detection settings
threshold = 0.8
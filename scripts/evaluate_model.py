import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

# Import from config
from src.config import (
    DATA_PATH,
    signs,
    no_sequences,
    sequence_length,
    MODEL_NAME,
    TEST_SIZE
)

# Import data utilities
from src.data_utils import (
    create_label_map,
    load_sequences,
    prepare_data
)

# evaluation code
if __name__ == "__main__":
    # Create label map
    label_map = create_label_map(signs)
    
    # Load sequences
    sequences, labels = load_sequences(DATA_PATH, signs, no_sequences, sequence_length, label_map)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(sequences, labels, test_size=TEST_SIZE)
    
    # Loading model 
    model = load_model(MODEL_NAME)
    
    # Predict
    result = model.predict(X_test)
    
    # Prepare metrics
    ytrue = np.argmax(y_test, axis=1).tolist()
    yfalse = np.argmax(result, axis=1).tolist()
    print("ytrue", ytrue)
    print("yfalse", yfalse)
    
    # Calculate accuracy
    accuracy = accuracy_score(ytrue, yfalse)
    print(f"Accuracy: {accuracy}")
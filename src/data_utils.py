import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def create_label_map(signs):
    return {label: num for num, label in enumerate(signs)}

def load_sequences(data_path, signs, no_sequences, sequence_length, label_map):
    sequences, labels = [], []
    for action in signs:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(data_path, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    return np.array(sequences), np.array(labels)

def prepare_data(sequences, labels, test_size=0.15):
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test
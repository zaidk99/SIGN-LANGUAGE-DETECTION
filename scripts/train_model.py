import os
from tensorflow.keras.callbacks import TensorBoard

# Import from config
from src.config import (
    DATA_PATH,
    signs,
    no_sequences,
    sequence_length,
    LOG_DIR,
    EPOCHS,
    TEST_SIZE
)

# Import data utilities
from src.data_utils import (
    create_label_map,
    load_sequences,
    prepare_data
)

# Import model utilities
from src.model_utils import (
    build_model,
    compile_model
)

# TensorBoard setup
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
tb_callback = TensorBoard(log_dir=LOG_DIR)

# training code
if __name__ == "__main__":
    # Create label map
    label_map = create_label_map(signs)
    
    # Load sequences (from Cell 6)
    sequences, labels = load_sequences(DATA_PATH, signs, no_sequences, sequence_length, label_map)
    print("sequences shape:", sequences.shape)
    print("labels shape:", labels.shape)
    
    
    X_train, X_test, y_train, y_test = prepare_data(sequences, labels, test_size=TEST_SIZE)
    print("x_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    
    model = build_model(sequence_length, 1662, len(signs))
    
    # Compile model
    model = compile_model(model)
    model.summary()
    
    model.fit(X_train, y_train, epochs=EPOCHS, callbacks=[tb_callback])
    
    # Save model 
    from src.config import MODEL_NAME
    model.save(MODEL_NAME)
    print(f"Model saved as {MODEL_NAME}")
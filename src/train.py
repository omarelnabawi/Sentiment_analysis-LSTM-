# ==============================
# Import Necessary Libraries
# ==============================
from .prepocessing import preprocess_data
from .model import build_model
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger

# ==============================
# Prepare Data
# ==============================
x_train, y_train, x_test, y_test, tokenizer = preprocess_data()

# ==============================
# Train Function
# ==============================
def model_train(i):

    # Create folders if not exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("models_metrics", exist_ok=True)
    os.makedirs("models_metrics/LOGS",exist_ok=True)
    os.makedirs("models_metrics/HISTORY",exist_ok=True)


    model_path = f"models/model_epoch({i}).keras"
    log_path = f"models_metrics/LOGS/training_log_epoch({i}).csv"

    # CSV Logger (يسجل كل epoch تلقائيًا)
    csv_logger = CSVLogger(log_path, append=True)

    # ==============================
    # Train or Load Model
    # ==============================
    if not os.path.exists(model_path):

        print("Training model...")

        model = build_model()

        history = model.fit(
            x_train,
            y_train,
            epochs=i,
            batch_size=64,
            validation_data=(x_test, y_test),
            callbacks=[csv_logger],
            verbose=1
        )

        # Save model
        model.save(model_path)

        # Save history separately
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(f"models_metrics/HISTORY/history_epoch({i}).csv", index=False)

    else:
        print("Loading existing model...")
        model = load_model(model_path)

    # ==============================
    # Evaluate
    # ==============================
    #loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    #print(f"\nTest Loss: {loss:.4f}")
    #print(f"Test Accuracy: {accuracy:.4f}")

    return model




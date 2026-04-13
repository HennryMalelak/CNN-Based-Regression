import chartjs

#!/usr/bin/env python3.10
import os
from typing import Tuple
# Suppress TensorFlow INFO messages (CPU optimization warning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Verify TensorFlow version and GPU usage
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Load data with error handling
try:
    train: pd.DataFrame = pd.read_parquet("data/train-00000-of-00001.parquet")
    test: pd.DataFrame = pd.read_parquet("data/test-00000-of-00001.parquet")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the data files exist in the 'data' directory.")
    exit(1)
except Exception as e:
    print(f"Error loading parquet files: {e}")
    exit(1)

# Prepare training data
num_dropped = len(train) - len(train.dropna(subset=['question', 'correct_answer']))
print(f"Dropped {num_dropped} rows due to missing values.")
train = train.dropna(subset=['question', 'correct_answer'])
train['input_text'] = train['question'] + " " + train['correct_answer']
train['duration'] = np.random.uniform(1, 10, size=len(train))  # Random durations [1, 10]

# Normalize target variable
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(train['duration'].values.reshape(-1, 1)).flatten()

# Prepare test data
test = test.dropna(subset=['question', 'correct_answer'])
test['input_text'] = test['question'] + " " + test['correct_answer']

# --- Ridge Model ---
print("Training Ridge Model...")
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X = vectorizer.fit_transform(train['input_text']).toarray()
X_test_ridge = vectorizer.transform(test['input_text']).toarray()

# Split data for Ridge
X_train_ridge, X_val_ridge, y_train_ridge, y_val_ridge = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler_X = StandardScaler()
X_train_ridge = scaler_X.fit_transform(X_train_ridge)
X_val_ridge = scaler_X.transform(X_val_ridge)
X_test_ridge = scaler_X.transform(X_test_ridge)

# Train Ridge model
ridge_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
ridge_model.fit(X_train_ridge, y_train_ridge)

# Evaluate Ridge model
y_pred_ridge = ridge_model.predict(X_val_ridge)
mse_ridge = mean_squared_error(y_val_ridge, y_pred_ridge)
mae_ridge = mean_absolute_error(y_val_ridge, y_pred_ridge)
print(f"Ridge Model - Validation Loss (MSE): {mse_ridge:.4f}, MAE: {mae_ridge:.4f}")

# --- CNN Model ---
print("\nTraining CNN Model...")
# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train['input_text'])
sequences = tokenizer.texts_to_sequences(train['input_text'])

# Set maxlen based on sequence length distribution
sequence_lengths = [len(seq) for seq in sequences]
maxlen = int(pd.Series(sequence_lengths).quantile(0.95))
print(f"Using maxlen={maxlen} based on sequence length distribution.")

# Pad sequences
padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
X_cnn = padded
y_cnn = y

# Prepare test data for CNN
test_sequences = tokenizer.texts_to_sequences(test['input_text'])
test_padded = pad_sequences(test_sequences, maxlen=maxlen, padding='post', truncating='post')
X_test_cnn = test_padded

# Split data for CNN
X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(
    X_cnn, y_cnn, test_size=0.2, random_state=42
)

# Build CNN model
def build_cnn_model(vocab_size: int) -> Sequential:
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Calculate vocab size
vocab_size = min(len(tokenizer.word_index) + 1, 10000)
cnn_model = build_cnn_model(vocab_size)
cnn_model.summary()

# Train CNN model for 20 epochs
history = cnn_model.fit(
    X_train_cnn, y_train_cnn,
    validation_data=(X_val_cnn, y_val_cnn),
    epochs=20, batch_size=32, verbose=1
)

# Evaluate CNN model
loss_cnn, mae_cnn = cnn_model.evaluate(X_val_cnn, y_val_cnn, verbose=0)
print(f"CNN Model - Validation Loss (MSE): {loss_cnn:.10f}, MAE: {mae_cnn:.4f}")

# --- Plot Training and Validation Loss ---
# Line chart for training and validation loss

{
  "type": "line",
  "data": {
    "labels": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "datasets": [
      {
        "label": "Training Loss",
        "data": history.history['loss'],
        "borderColor": "#1f77b4",
        "backgroundColor": "rgba(31, 119, 180, 0.1)",
        "fill": True,
        "tension": 0.4
      },
      {
        "label": "Validation Loss",
        "data": history.history['val_loss'],
        "borderColor": "#ff7f0e",
        "backgroundColor": "rgba(255, 127, 14, 0.1)",
        "fill": True,
        "tension": 0.4
      }
    ]
  },
  "options": {
    "responsive": True,
    "plugins": {
      "title": {
        "display": True,
        "text": "CNN Model Training and Validation Loss per Epoch"
      },
      "legend": {
        "position": "top"
      }
    },
    "scales": {
      "x": {
        "title": {
          "display": True,
          "text": "Epoch"
        }
      },
      "y": {
        "title": {
          "display": True,
          "text": "Loss (MSE)"
        },
        "beginAtZero": True
      }
    }
  }
}

# --- Example Predictions ---
# CNN predictions on test set
y_pred_test_cnn = cnn_model.predict(X_test_cnn).flatten()

# Inverse transform predictions to original duration scale
y_pred_test_cnn_orig = scaler_y.inverse_transform(y_pred_test_cnn.reshape(-1, 1)).flatten()

# Display 5 example predictions with original text
print("\nExample CNN Predictions on Test Set (with Original Text):")
for i in range(min(5, len(test))):
    print(f"\nSample {i+1}:")
    print(f"Input Text: {test['input_text'].iloc[i]}")
    print(f"Predicted Duration: {y_pred_test_cnn_orig[i]:.4f} seconds")

# --- Evaluate on Test Set ---
# Ridge predictions
y_pred_test_ridge = ridge_model.predict(X_test_ridge)

# Display 5 example predictions with original text
print("\nExample Ridge predictions on Test Set (with Original Text):")
for i in range(min(5, len(test))):
    print(f"\nSample {i+1}:")
    print(f"Input Text: {test['input_text'].iloc[i]}")
    print(f"Predicted Duration: {y_pred_test_ridge[i]:.4f} seconds")

# Print first 5 predictions (normalized scale)
print("\nTest Set Predictions (first 5 samples, normalized scale):")
print("Ridge Model Predictions:", y_pred_test_ridge[:5])
print("CNN Model Predictions:", y_pred_test_cnn[:5])

# --- Compare Models ---
print("\nModel Comparison (Validation Set):")
print(f"Ridge Model - MSE: {mse_ridge:.10f}, MAE: {mae_ridge:.4f}")
print(f"CNN Model - MSE: {loss_cnn:.10f}, MAE: {mae_cnn:.4f}")

import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
cleaned_csv_path = os.path.join(
    script_dir, "..", "data", "processed", "cleaned_metadata.csv")
image_dir = os.path.join(script_dir, "..", "data", "raw", "images")

# Load cleaned metadata
df = pd.read_csv(cleaned_csv_path)
df['file_path'] = df['image_id'].apply(
    lambda x: os.path.join(image_dir, f"image_{x}.jpg"))

# Map labels to binary integers (REAL = 0, FAKE = 1)
df['target'] = df['label'].apply(lambda x: 1 if x == 'FAKE' else 0)

# Split into train (80%) and validation (20%) sets
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['target'])

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


def load_and_preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img, label


# Create TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_df['file_path'].values, train_df['target'].values))
train_dataset = train_dataset.map(
    load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1000).batch(
    BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_df['file_path'].values, val_df['target'].values))
val_dataset = val_dataset.map(
    load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Build a Convolutional Neural Network (CNN) model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification output
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

model.summary()

# Train the model for a baseline run (e.g., 5 epochs)
print("\nStarting model training...")
history = model.fit(train_dataset, validation_data=val_dataset, epochs=5)
print("Model training completed successfully!")

# Define model output directory
models_dir = os.path.join(script_dir, "..", "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "face_authenticity_cnn.keras")

# Save the trained model
model.save(model_path)
print(f"\nModel successfully saved to: {model_path}")

# Evaluate final performance metrics on the validation/test set
val_loss, val_acc, val_precision, val_recall = model.evaluate(val_dataset)
val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)

print("\n--- Final Capstone Evaluation Metrics ---")
print(f"Validation Accuracy:  {val_acc * 100:.2f}%")
print(f"Precision:            {val_precision * 100:.2f}%")
print(f"Recall:               {val_recall * 100:.2f}%")
print(f"F1-Score:             {val_f1 * 100:.2f}%")

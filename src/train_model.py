import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

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

# Use predefined dataset splits ('train', 'val', 'test') instead of train_test_split()
split_col = df['dataset_split'].astype(str).str.strip().str.lower()
train_df = df[split_col == 'train'].copy()
val_df = df[split_col == 'val'].copy()
test_df = df[split_col == 'test'].copy()

print(f"Data Splits Loaded:")
print(f"  Training samples:   {len(train_df)}")
print(f"  Validation samples: {len(val_df)}")
print(f"  Testing samples:    {len(test_df)}")

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

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_df['file_path'].values, test_df['target'].values))
test_dataset = test_dataset.map(
    load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

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

# Train the model for 5 epochs
print("\nStarting model training...")
history = model.fit(train_dataset, validation_data=val_dataset, epochs=5)
print("Model training completed successfully!")

# Define output directories
docs_dir = os.path.join(script_dir, "..", "docs")
reports_fig_dir = os.path.join(script_dir, "..", "reports", "figures")
models_dir = os.path.join(script_dir, "..", "models")
os.makedirs(docs_dir, exist_ok=True)
os.makedirs(reports_fig_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Generate and save training curves
epochs_range = range(1, len(history.history['accuracy']) + 1)
plt.figure(figsize=(12, 5))

# Accuracy Subplot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, [acc * 100 for acc in history.history['accuracy']], 'o-', label='Training Accuracy', color='#2563EB', linewidth=2)
plt.plot(epochs_range, [acc * 100 for acc in history.history['val_accuracy']], 's-', label='Validation Accuracy', color='#10B981', linewidth=2)
plt.title('Model Accuracy vs Epochs', fontsize=13, fontweight='bold', pad=10)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Accuracy (%)', fontsize=11)
plt.legend(loc='lower right', frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)

# Loss Subplot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], 'o-', label='Training Loss', color='#DC2626', linewidth=2)
plt.plot(epochs_range, history.history['val_loss'], 's-', label='Validation Loss', color='#F59E0B', linewidth=2)
plt.title('Model Loss vs Epochs', fontsize=13, fontweight='bold', pad=10)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Loss (Binary Crossentropy)', fontsize=11)
plt.legend(loc='upper right', frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
curves_path = os.path.join(docs_dir, "training_curves.png")
plt.savefig(curves_path, dpi=300)
plt.savefig(os.path.join(reports_fig_dir, "training_curves.png"), dpi=300)
plt.close()
print(f"Training curves saved to: {curves_path}")

# Save the trained model
model_path = os.path.join(models_dir, "face_authenticity_cnn.keras")
model.save(model_path)
print(f"\nModel successfully saved to: {model_path}")

# Evaluate final performance metrics on validation set
val_loss, val_acc, val_precision, val_recall = model.evaluate(val_dataset)
val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)

# Evaluate final performance metrics on true test set
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_dataset)
test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)

print("\n--- Validation Set Metrics ---")
print(f"Validation Accuracy:  {val_acc * 100:.2f}%")
print(f"Precision:            {val_precision * 100:.2f}%")
print(f"Recall:               {val_recall * 100:.2f}%")
print(f"F1-Score:             {val_f1 * 100:.2f}%")

print("\n--- Predefined Test Set Metrics ---")
print(f"Test Accuracy:        {test_acc * 100:.2f}%")
print(f"Precision:            {test_precision * 100:.2f}%")
print(f"Recall:               {test_recall * 100:.2f}%")
print(f"F1-Score:             {test_f1 * 100:.2f}%")

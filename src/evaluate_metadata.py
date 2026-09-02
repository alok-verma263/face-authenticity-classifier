import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
cleaned_csv_path = os.path.join(
    script_dir, "..", "data", "processed", "cleaned_metadata.csv")
image_dir = os.path.join(script_dir, "..", "data", "raw", "images")
model_path = os.path.join(script_dir, "..", "models",
                          "face_authenticity_cnn.keras")

# Load model and data
model = tf.keras.models.load_model(model_path)
df = pd.read_csv(cleaned_csv_path)
df['file_path'] = df['image_id'].apply(
    lambda x: os.path.join(image_dir, f"image_{x}.jpg"))
df['target'] = df['label'].apply(lambda x: 1 if x == 'FAKE' else 0)

# Re-create the exact same validation split used during training (random_state=42, stratify)
_, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['target'])

IMG_SIZE = (128, 128)


def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img / 255.0


# Load validation images into a numpy array for direct slice evaluation
print("Loading validation images into memory for metadata evaluation...")
val_images = np.array([load_image(fp).numpy() for fp in val_df['file_path']])
val_targets = val_df['target'].values

# Predict probabilities
preds = model.predict(val_images)
val_df['predicted_prob'] = preds.flatten()
val_df['predicted_target'] = (val_df['predicted_prob'] >= 0.5).astype(int)

# Analyze performance by available metadata attributes (e.g., gender, detection difficulty)
print("\n--- Performance Breakdown by Metadata Attributes ---")

for attr in ['gender', 'detection_difficulty']:
    if attr in val_df.columns:
        print(f"\nBreakdown by: {attr.upper()}")
        for val, group in val_df.groupby(attr):
            if len(group) > 0:
                acc = (group['target'] == group['predicted_target']).mean()
                print(
                    f"  - {val} (Count: {len(group)}): Accuracy = {acc * 100:.2f}%")

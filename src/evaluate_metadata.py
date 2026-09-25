import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
cleaned_csv_path = os.path.join(
    script_dir, "..", "data", "processed", "cleaned_metadata.csv")
image_dir = os.path.join(script_dir, "..", "data", "raw", "images")
model_path = os.path.join(script_dir, "..", "models",
                          "face_authenticity_cnn.keras")
docs_dir = os.path.join(script_dir, "..", "docs")
reports_fig_dir = os.path.join(script_dir, "..", "reports", "figures")
os.makedirs(docs_dir, exist_ok=True)
os.makedirs(reports_fig_dir, exist_ok=True)

# Load model and data
print("Loading trained model and metadata...")
model = tf.keras.models.load_model(model_path)
df = pd.read_csv(cleaned_csv_path)
df['file_path'] = df['image_id'].apply(
    lambda x: os.path.join(image_dir, f"image_{x}.jpg"))
df['target'] = df['label'].apply(lambda x: 1 if x == 'FAKE' else 0)

# Evaluate exclusively on the predefined Testing split ('test')
split_col = df['dataset_split'].astype(str).str.strip().str.lower()
test_df = df[split_col == 'test'].copy().reset_index(drop=True)

print(f"Total Predefined Test Set Samples: {len(test_df)}")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img / 255.0


# Build optimized TensorFlow dataset for batch inference
print("Running batch inference on Test Set...")
test_ds = tf.data.Dataset.from_tensor_slices(test_df['file_path'].values)
test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)

# Predict probabilities and assign classes
preds = model.predict(test_ds)
test_df['predicted_prob'] = preds.flatten()
test_df['predicted_target'] = (test_df['predicted_prob'] >= 0.5).astype(int)

# Overall Test Performance Metrics
y_true = test_df['target'].values
y_pred = test_df['predicted_target'].values

overall_acc = (y_true == y_pred).mean()
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='binary', zero_division=0)

print("\n" + "=" * 55)
print("     OFFICIAL PREDEFINED TEST SET EVALUATION")
print("=" * 55)
print(f"Total Test Samples:     {len(test_df)}")
print(f"Test Accuracy:          {overall_acc * 100:.2f}%")
print(f"Precision:              {precision * 100:.2f}%")
print(f"Recall:                 {recall * 100:.2f}%")
print(f"F1-Score:               {f1 * 100:.2f}%")
print("=" * 55)

print("\nClassification Report (Test Set):")
print(classification_report(y_true, y_pred, target_names=['REAL (0)', 'FAKE (1)'], digits=4))

# Generate and Save Confusion Matrix
print("Generating Confusion Matrix plot...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['REAL', 'FAKE'],
            yticklabels=['REAL', 'FAKE'],
            annot_kws={"size": 14, "weight": "bold"})
plt.title('Confusion Matrix - Test Set Evaluation', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.tight_layout()

cm_path = os.path.join(docs_dir, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.savefig(os.path.join(reports_fig_dir, "confusion_matrix.png"), dpi=300)
plt.close()
print(f"Confusion matrix saved to: {cm_path}")

# Demographic and Metadata Robustness Breakdown
print("\n" + "=" * 70)
print("       PERFORMANCE BREAKDOWN BY DEMOGRAPHIC & IMAGE METADATA")
print("=" * 70)

metadata_attributes = [
    ('gender', 'Gender Demographics'),
    ('age_group', 'Age Group Demographics'),
    ('image_quality', 'Image Quality Levels'),
    ('detection_difficulty', 'Detection Difficulty Tiers')
]

for attr, title in metadata_attributes:
    if attr in test_df.columns:
        print(f"\n[{title.upper()}] (Attribute: '{attr}')")
        print(f"{'Category':<20} | {'Count':<8} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
        print("-" * 75)
        for val, group in test_df.groupby(attr):
            if len(group) > 0:
                g_true = group['target'].values
                g_pred = group['predicted_target'].values
                g_acc = (g_true == g_pred).mean() * 100
                g_prec, g_rec, g_f1, _ = precision_recall_fscore_support(
                    g_true, g_pred, average='binary', zero_division=0)
                print(f"{str(val):<20} | {len(group):<8} | {g_acc:>9.2f}% | {g_prec * 100:>9.2f}% | {g_rec * 100:>9.2f}% | {g_f1 * 100:>9.2f}%")
print("\n" + "=" * 70)

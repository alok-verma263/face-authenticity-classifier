import os
import pandas as pd
from PIL import Image

# Define paths relative to src/
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "data", "raw",
                        "AI_Classification-Project.csv")
image_dir = os.path.join(script_dir, "..", "data", "raw", "images")
processed_dir = os.path.join(script_dir, "..", "data", "processed")

# Create processed folder
os.makedirs(processed_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(csv_path)
print(f"Original dataset size: {len(df)} rows")

# Filter rows to only keep records where the local image file actually exists
valid_rows = []
for index, row in df.iterrows():
    img_id = row['image_id']
    file_path = os.path.join(image_dir, f"image_{img_id}.jpg")
    if os.path.exists(file_path):
        valid_rows.append(row)

# Create a clean filtered DataFrame
df_clean = pd.DataFrame(valid_rows)
print(
    f"Filtered dataset size (matching downloaded images): {len(df_clean)} rows")

# Save the cleaned metadata for modeling
cleaned_csv_path = os.path.join(processed_dir, "cleaned_metadata.csv")
df_clean.to_csv(cleaned_csv_path, index=False)
print(f"Cleaned metadata saved to {cleaned_csv_path}")

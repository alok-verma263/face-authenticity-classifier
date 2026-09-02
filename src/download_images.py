import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

# Get the directory where this script is located (src/)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths relative to the project root
csv_path = os.path.join(script_dir, "..", "data", "raw",
                        "AI_Classification-Project.csv")
image_dir = os.path.join(script_dir, "..", "data", "raw", "images")

# Create image storage directory if it doesn't exist
os.makedirs(image_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows. Starting image download pipeline...")

# Loop through each row to download the image
success_count = 0
fail_count = 0

for index, row in df.iterrows():
    img_id = row['image_id']
    img_url = row['image_url']
    file_path = os.path.join(image_dir, f"image_{img_id}.jpg")

    # Skip if already downloaded
    if os.path.exists(file_path):
        success_count += 1
        continue

    try:
        response = requests.get(img_url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(file_path, "JPEG")
            success_count += 1
        else:
            fail_count += 1
    except Exception as e:
        fail_count += 1

print(
    f"\nDownload complete! Successful: {success_count}, Failed/Skipped: {fail_count}")

"""
AI-Based Authentic Face Image Verification System
Proxy script allowing execution via: python src/predict.py <image_path>
"""
import os
import sys

# Ensure root directory is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from predict import main, verify_image

if __name__ == "__main__":
    main()

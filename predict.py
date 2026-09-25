"""
AI-Based Authentic Face Image Verification System
Single Image Prediction and Authenticity Verification Script

Usage:
    python predict.py <path_to_image>
    python predict.py --image <path_to_image>
    python predict.py --help
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf

# Suppress TensorFlow logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_SIZE = (128, 128)


def get_default_model_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "face_authenticity_cnn.keras")
    if not os.path.exists(model_path):
        # Also check inside parent if running from src
        parent_model_path = os.path.join(script_dir, "..", "models", "face_authenticity_cnn.keras")
        if os.path.exists(parent_model_path):
            return os.path.abspath(parent_model_path)
    return os.path.abspath(model_path)


def load_and_preprocess_single_image(image_path):
    """Loads and preprocesses an image file for CNN inference."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    img_raw = tf.io.read_file(image_path)
    # Decode image (supports JPEG, PNG, etc.)
    img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
    img = tf.expand_dims(img, axis=0)       # Add batch dimension: (1, 128, 128, 3)
    return img


def verify_image(image_path, model_path=None):
    """
    Evaluates a single facial image and returns authenticity classification
    along with confidence score percentage.
    """
    if model_path is None:
        model_path = get_default_model_path()

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}. "
            "Please train the model first by running: python src/train_model.py"
        )

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Preprocess image
    img_tensor = load_and_preprocess_single_image(image_path)

    # Predict probability (target: 0 = REAL, 1 = FAKE)
    prob_fake = float(model.predict(img_tensor, verbose=0)[0][0])
    prob_real = 1.0 - prob_fake

    if prob_fake >= 0.5:
        prediction = "FAKE"
        confidence = prob_fake * 100.0
        status = "Manipulated / AI-Generated Face"
        is_authentic = False
    else:
        prediction = "REAL"
        confidence = prob_real * 100.0
        status = "Authentic Face"
        is_authentic = True

    return {
        "image_path": os.path.abspath(image_path),
        "prediction": prediction,
        "confidence_percentage": round(confidence, 2),
        "status": status,
        "is_authentic": is_authentic,
        "prob_real": round(prob_real, 4),
        "prob_fake": round(prob_fake, 4)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify face authenticity (REAL vs FAKE) with Confidence Score."
    )
    parser.add_argument(
        "image_pos",
        nargs="?",
        default=None,
        help="Path to the image to verify (optional positional argument)"
    )
    parser.add_argument(
        "-i", "--image",
        dest="image_flag",
        default=None,
        help="Path to the image to verify"
    )
    parser.add_argument(
        "-m", "--model",
        dest="model_path",
        default=None,
        help="Custom path to saved .keras model"
    )

    args = parser.parse_args()

    # Determine input image path
    target_image = args.image_flag or args.image_pos

    if not target_image:
        # Check for a sample image in data/raw/images as demonstration
        sample_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "raw", "images", "image_1.jpg"
        )
        if os.path.exists(sample_path):
            print("No image path provided. Running verification on sample image:")
            print(f"  --> {sample_path}")
            print("Tip: You can pass your own image with: python predict.py path/to/face.jpg\n")
            target_image = sample_path
        else:
            print("Error: No image path provided.")
            print("Usage: python predict.py <path_to_image>")
            sys.exit(1)

    try:
        result = verify_image(target_image, args.model_path)

        print("\n" + "=" * 60)
        print("    AI FACE AUTHENTICITY VERIFICATION RESULT")
        print("=" * 60)
        print(f"Target Image:      {result['image_path']}")
        print(f"Authenticity:      {result['prediction']} ({result['status']})")
        print(f"Confidence Score:  {result['confidence_percentage']:.2f}%")
        print("-" * 60)
        print(f"Authentic Prob:    {result['prob_real'] * 100:.2f}%")
        print(f"Manipulated Prob:  {result['prob_fake'] * 100:.2f}%")
        print("=" * 60)
        if result['is_authentic']:
            print("Status: [PASS] Image verified as an authentic human face.\n")
        else:
            print("Status: [ALERT] Image flagged as synthetic / deepfake.\n")

    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

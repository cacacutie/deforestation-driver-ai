"""Predict the dataset class for one RGB satellite image."""
from pathlib import Path
import argparse
import json
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "deforestation_cnn.keras"
CLASS_PATH = ROOT / "outputs" / "class_names.json"
IMG_SIZE = (128, 128)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to an RGB image file")
    args = parser.parse_args()

    if not MODEL_PATH.exists() or not CLASS_PATH.exists():
        raise FileNotFoundError(
            "Model belum ada. Jalankan python src/train.py terlebih dahulu."
        )

    with open(CLASS_PATH, encoding="utf-8") as f:
        class_names = json.load(f)

    image = tf.keras.utils.load_img(args.image, target_size=IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)

    model = tf.keras.models.load_model(MODEL_PATH)
    scores = model.predict(array, verbose=0)[0]
    order = np.argsort(scores)[::-1]

    print("Prediksi kategori (bukan bukti penyebab di lapangan):")
    for idx in order[:min(3, len(class_names))]:
        print(f"- {class_names[idx]}: {scores[idx]:.2%}")

if __name__ == "__main__":
    main()

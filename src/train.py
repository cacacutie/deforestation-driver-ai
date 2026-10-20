# Temporary file

from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

# Konfigurasi proyek
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42


def load_dataset(split, class_names=None, shuffle=False):
    folder = DATA_DIR / split

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder {folder} tidak ditemukan. "
            "Siapkan dataset train, val, dan test terlebih dahulu."
        )

    return tf.keras.utils.image_dataset_from_directory(
        folder,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        class_names=class_names,
        shuffle=shuffle,
        seed=SEED
    )


def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),

        tf.keras.layers.Conv2D(
            32, 3, activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            64, 3, activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            128, 3, activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(
            num_classes, activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Membaca data latih dan nama kelas
    train_ds = load_dataset("train", shuffle=True)
    class_names = train_ds.class_names

    # Data validasi dan data pengujian terpisah
    val_ds = load_dataset("val", class_names)
    test_ds = load_dataset("test", class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    with open(
        OUTPUT_DIR / "class_names.json",
        "w", encoding="utf-8"
    ) as file:
        json.dump(class_names, file, indent=2)

    print("Kategori:", class_names)

    model = build_model(len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "deforestation_cnn.keras"),
            monitor="val_loss",
            save_best_only=True
        )
    ]

    # Pelatihan CNN
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Grafik akurasi
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(
        history.history["val_accuracy"],
        label="Validation"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Model Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy.png")
    plt.close()

    # Grafik loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(
        history.history["val_loss"],
        label="Validation"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Model Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss.png")
    plt.close()

    # Evaluasi akhir menggunakan data test
    loss, accuracy = model.evaluate(test_ds, verbose=0)

    y_true = np.concatenate([
        labels.numpy() for _, labels in test_ds
    ])

    probabilities = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    print(f"\nTest accuracy: {accuracy:.2%}")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    ))

    with open(
        OUTPUT_DIR / "classification_report.json",
        "w", encoding="utf-8"
    ) as file:
        json.dump({
            "test_loss": float(loss),
            "test_accuracy": float(accuracy),
            "report": report
        }, file, indent=2)

    # Confusion matrix
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    from sklearn.metrics import ConfusionMatrixDisplay

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    display.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()

    print("\nPelatihan dan evaluasi selesai.")
    print("Hasil tersimpan di folder outputs/ dan models/")


if __name__ == "__main__":
    main()

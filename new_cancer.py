"""
Skin Cancer Detection using Deep Learning (CNN)
==================================================
Classifies a skin lesion image as BENIGN or MALIGNANT (cancerous).

Dataset: HAM10000 ("Human Against Machine with 10000 training images")
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
  - ~10,015 dermatoscopic images across 7 lesion types:
      Malignant : mel (melanoma), bcc (basal cell carcinoma), akiec (actinic keratosis)
      Benign    : nv (melanocytic nevus), bkl (benign keratosis), df (dermatofibroma), vasc (vascular lesion)

This script generates a realistic SYNTHETIC image dataset (colored lesion-like
blobs) so it runs immediately with no downloads. Swap load_data() for real
HAM10000 images + metadata.csv when you're ready (instructions at the bottom).

Once trained, use predict_image("your_skin_photo.jpg") to test on a real photo.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    tf = None
    layers = None
    models = None
    TF_AVAILABLE = False

# Fallback ML model (when TF is not installed)
try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
except Exception:
    joblib = None
    RandomForestClassifier = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

IMG_SIZE = 64  # resize all images to 64x64 for a fast, simple CNN
np.random.seed(42)
if TF_AVAILABLE:
    tf.random.set_seed(42)

# ---- SWITCH BETWEEN SYNTHETIC DEMO DATA AND REAL HAM10000 DATA HERE ----
USE_REAL_DATA = True  # set to True once you've downloaded HAM10000 (see steps below)
METADATA_PATH = "HAM10000_metadata.csv"
IMAGES_DIRS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]  # both image folders
MALIGNANT_TYPES = {"mel", "bcc", "akiec"}  # everything else counts as benign


# -----------------------------------------------------------------------
# 1. CREATE / LOAD DATA
# -----------------------------------------------------------------------
def load_real_ham10000_data():
    """Loads the real HAM10000 dataset (metadata CSV + image folders).
    Only used when USE_REAL_DATA = True. See the step-by-step setup guide
    for how to download and place these files."""
    import pandas as pd
    import os

    meta = pd.read_csv(METADATA_PATH)

    # HAM10000 images are split across two folders - build a lookup so we
    # don't care which folder a given image_id actually lives in
    image_lookup = {}
    for folder in IMAGES_DIRS:
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_id = os.path.splitext(fname)[0]
                image_lookup[image_id] = os.path.join(folder, fname)

    X, y = [], []
    skipped = 0
    for _, row in meta.iterrows():
        image_id = row["image_id"]
        path = image_lookup.get(image_id)
        if path is None:
            skipped += 1
            continue
        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        X.append(np.array(img) / 255.0)
        y.append(1 if row["dx"] in MALIGNANT_TYPES else 0)

    if skipped:
        print(f"Warning: {skipped} images listed in metadata were not found on disk.")

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int32")
    return X, y



def generate_synthetic_lesion(malignant: bool):
    """Draws a simple lesion-like blob. Malignant ones get irregular
    borders + darker/mixed colors; benign ones are round and even-colored
    (loosely mimicking the real visual cues dermatologists use: the
    'ABCDE' rule — Asymmetry, Border, Color, Diameter, Evolving)."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (200 + np.random.randint(-20, 20),
                                                     170 + np.random.randint(-20, 20),
                                                     150 + np.random.randint(-20, 20)))
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2

    if malignant:
        # irregular, asymmetric blob with mixed dark colors
        base_color = (np.random.randint(30, 80), np.random.randint(10, 40), np.random.randint(10, 40))
        points = []
        for angle in range(0, 360, 20):
            r = np.random.randint(10, 22)  # irregular radius -> jagged border
            x = cx + r * np.cos(np.radians(angle))
            y = cy + r * np.sin(np.radians(angle))
            points.append((x, y))
        draw.polygon(points, fill=base_color)
        # add a patch of a second color (color variation is a malignancy cue)
        draw.ellipse([cx - 6, cy - 6, cx + 10, cy + 10],
                     fill=(base_color[0] + 40, base_color[1], base_color[2]))
    else:
        # smooth, symmetric, single-color mole
        r = np.random.randint(12, 18)
        color = (np.random.randint(90, 140), np.random.randint(60, 90), np.random.randint(50, 80))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    return np.array(img) / 255.0  # normalize to [0,1]


def load_data(n_per_class=300):
    X, y = [], []
    for label, malignant in [(0, False), (1, True)]:  # 0=benign, 1=malignant
        for _ in range(n_per_class):
            X.append(generate_synthetic_lesion(malignant))
            y.append(label)
    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int32")
    return X, y


# -----------------------------------------------------------------------
# 2. BUILD CNN MODEL
# -----------------------------------------------------------------------
def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),  # binary: benign(0) vs malignant(1)
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_sklearn_model():
    """Return an untrained scikit-learn RandomForest classifier as a fallback."""
    if RandomForestClassifier is None:
        raise RuntimeError("scikit-learn is required for the fallback model but is not available.")
    return RandomForestClassifier(n_estimators=200, random_state=42)


# -----------------------------------------------------------------------
# 3. MAIN: TRAIN + EVALUATE
# -----------------------------------------------------------------------
CLASS_NAMES = ["Benign", "Malignant"]
model = None  # trained model kept here for predict_image() to reuse
TF_MODEL_PATH = "skin_cancer_cnn_model.keras"
SK_MODEL_PATH = "skin_cancer_rf.joblib"


def main():
    global model

    if USE_REAL_DATA:
        print("Loading real HAM10000 dataset...")
        X, y = load_real_ham10000_data()
        epochs = 20  # real data needs more epochs than the synthetic demo
    else:
        print("Generating synthetic demo dataset...")
        X, y = load_data(n_per_class=300)
        epochs = 10

    print(f"Dataset shape: {X.shape}, Labels: {np.bincount(y)} (benign, malignant)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Real HAM10000 is imbalanced (~67% benign nv). Class weights tell the
    # model to pay more attention to the minority (malignant) class instead
    # of just learning to predict "benign" every time.
    class_weight = None
    if USE_REAL_DATA:
        n_benign, n_malignant = np.bincount(y_train)
        total = n_benign + n_malignant
        class_weight = {
            0: total / (2 * n_benign),
            1: total / (2 * n_malignant),
        }
        print(f"Class weights (to correct imbalance): {class_weight}")

    # Build and train either a TensorFlow CNN or a scikit-learn fallback
    if TF_AVAILABLE:
        print("\nBuilding CNN...")
        model = build_cnn()
        model.summary()

        print("\nTraining...")
        history = model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=epochs,
            batch_size=32,
            class_weight=class_weight,
            verbose=2,
        )

        # Evaluate
        probs = model.predict(X_test, verbose=0).ravel()
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        print(f"\nTest Accuracy: {acc:.2%}")

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap="Reds")
        plt.title(f"Skin Cancer Detection - Test Accuracy: {acc:.1%}")
        plt.tight_layout()
        plt.savefig("skin_cancer_results.png", dpi=150)
        print("Saved confusion matrix -> skin_cancer_results.png")

        # Training curves
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="train")
        plt.plot(history.history["val_accuracy"], label="val")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="val")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_curves.png", dpi=150)
        print("Saved training curves -> training_curves.png")

        model.save(TF_MODEL_PATH)
        print(f"Saved model -> {TF_MODEL_PATH}")
    else:
        print("\nTensorFlow not available — training scikit-learn RandomForest fallback.")
        model = build_sklearn_model()
        # Flatten images for scikit-learn
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        model.fit(X_train_flat, y_train)
        preds = model.predict(X_test_flat)
        acc = accuracy_score(y_test, preds)
        print(f"\nTest Accuracy (RF): {acc:.2%}")

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap="Reds")
        plt.title(f"Skin Cancer Detection (RF) - Test Accuracy: {acc:.1%}")
        plt.tight_layout()
        plt.savefig("skin_cancer_results.png", dpi=150)
        print("Saved confusion matrix -> skin_cancer_results.png")

        # Save sklearn model
        if joblib is not None:
            joblib.dump(model, SK_MODEL_PATH)
            print(f"Saved model -> {SK_MODEL_PATH}")
        else:
            print("Warning: joblib not available; sklearn model not saved to disk.")


# -----------------------------------------------------------------------
# 4. PREDICT ON A REAL UPLOADED IMAGE
# -----------------------------------------------------------------------
def predict_image(image_path):
    """Loads a real skin photo, resizes it, and returns the model's prediction.
    Usage: predict_image("my_mole_photo.jpg")
    """
    global model
    # Load model if not already loaded
    if model is None:
        if TF_AVAILABLE and os.path.exists(TF_MODEL_PATH):
            model = tf.keras.models.load_model(TF_MODEL_PATH)
        elif os.path.exists(SK_MODEL_PATH) and joblib is not None:
            model = joblib.load(SK_MODEL_PATH)
        else:
            raise RuntimeError("No trained model found. Run the script to train a model first.")

    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0

    if TF_AVAILABLE and hasattr(model, "predict") and getattr(model, "save", None) is not None:
        inp = np.expand_dims(arr, axis=0)
        prob_malignant = model.predict(inp, verbose=0)[0][0]
    else:
        # sklearn model: flatten and use predict_proba if available
        flat = arr.reshape(1, -1)
        if hasattr(model, "predict_proba"):
            prob_malignant = model.predict_proba(flat)[0][1]
        else:
            prob = model.predict(flat)[0]
            prob_malignant = float(prob)

    label = CLASS_NAMES[int(prob_malignant > 0.5)]
    print(f"Prediction: {label} (confidence: {max(prob_malignant, 1 - prob_malignant):.1%})")
    return label, prob_malignant


def get_image_path_from_user():
    """Opens a file-picker window so the user can 'upload' (select) a skin
    photo from their computer. Falls back to typing a path if no display
    is available (e.g. running in a plain terminal/server)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()          # hide the empty root window
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select a skin photo to check",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        root.destroy()
        return path if path else None
    except Exception:
        # No GUI available -> ask for a path in the terminal instead
        path = input("Enter the full path to your skin photo (.jpg/.png): ").strip()
        return path if path else None


def show_prediction(image_path, label, prob_malignant):
    """Displays the uploaded image alongside the model's verdict."""
    img = Image.open(image_path).convert("RGB")
    confidence = max(prob_malignant, 1 - prob_malignant)
    color = "red" if label == "Malignant" else "green"

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prediction: {label}  ({confidence:.1%} confidence)", color=color, fontsize=13)
    plt.tight_layout()
    plt.savefig("your_result.png", dpi=150)
    plt.show()
    print(f"Saved annotated result -> your_result.png")


def run_interactive_check():
    """The main user-facing loop: ask for a photo, predict, repeat."""
    print("\n" + "=" * 60)
    print("SKIN CANCER DETECTION - Upload a photo to check")
    print("=" * 60)
    print("NOTE: This is a student ML project demo, NOT a medical")
    print("device. It cannot diagnose cancer. If you're worried")
    print("about a mole or skin change, please see a dermatologist.")
    print("=" * 60)

    while True:
        answer = input("\nWould you like to upload a skin photo to check? (y/n): ").strip().lower()
        if answer != "y":
            print("Okay, exiting. Stay healthy!")
            break

        image_path = get_image_path_from_user()
        if not image_path:
            print("No image selected/provided. Try again.")
            continue

        try:
            label, prob_malignant = predict_image(image_path)
            show_prediction(image_path, label, prob_malignant)

            if label == "Malignant":
                print("Result: Signs consistent with MALIGNANT lesion detected.")
                print("Please consult a dermatologist for a proper diagnosis.")
            else:
                print("Result: Lesion appears BENIGN based on this model.")
                print("Still, monitor any changes and consult a doctor if unsure.")
        except Exception as e:
            print(f"Couldn't process that image ({e}). Please try another file.")


if __name__ == "__main__":
    main()                    # trains the model (or you can skip this once
                              # skin_cancer_cnn_model.keras already exists)
    run_interactive_check()   # then asks the user to upload a photo to check


# -----------------------------------------------------------------------
# HOW TO USE THE REAL HAM10000 DATASET INSTEAD OF SYNTHETIC DATA
# -----------------------------------------------------------------------
# 1. Download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
#    You'll get: HAM10000_images_part_1/, HAM10000_images_part_2/, HAM10000_metadata.csv
# 2. Replace load_data() with something like:
#
#    import pandas as pd, os
#    def load_data():
#        meta = pd.read_csv("HAM10000_metadata.csv")
#        malignant_types = {"mel", "bcc", "akiec"}
#        X, y = [], []
#        for _, row in meta.iterrows():
#            path = os.path.join("images", row["image_id"] + ".jpg")
#            img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
#            X.append(np.array(img) / 255.0)
#            y.append(1 if row["dx"] in malignant_types else 0)
#        return np.array(X, dtype="float32"), np.array(y, dtype="int32")
#
# Everything else (CNN, training, evaluation, predict_image) stays the same.

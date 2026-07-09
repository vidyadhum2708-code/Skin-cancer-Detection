# Skin Cancer Detection Using Deep Learning

A Convolutional Neural Network (CNN) that classifies dermatoscopic skin lesion images as **Benign** or **Malignant**, built as part of an internship in AI \& Machine Learning in Computational Biology at [BioTechTrek](https://biotechtrek.in).

> ⚠️ **Disclaimer:** This is a student/academic project, **not** a certified medical device. It is not intended for real-world diagnostic use. 

---

## 📋 Overview

Skin cancer, when caught early, is highly treatable — but early detection depends on visual screening. This project explores whether a deep learning model can act as a supportive screening aid by classifying dermatoscopic images of skin lesions.

The model is trained on the **HAM10000** dataset ("Human Against Machine with 10000 training images"), grouping its 7 original diagnostic categories into two clinically relevant classes:

* **Malignant** → melanoma (mel), basal cell carcinoma (bcc), actinic keratosis (akiec)
* **Benign** → melanocytic nevus (nv), benign keratosis (bkl), dermatofibroma (df), vascular lesion (vasc)

## 🧠 Model Architecture

A CNN built with TensorFlow/Keras:

|Layer|Configuration|
|-|-|
|Input|64 × 64 × 3 (RGB)|
|Conv2D + ReLU|16 filters, 3×3|
|MaxPooling2D|2×2|
|Conv2D + ReLU|32 filters, 3×3|
|MaxPooling2D|2×2|
|Conv2D + ReLU|64 filters, 3×3|
|MaxPooling2D|2×2|
|Flatten|–|
|Dense + ReLU|64 units|
|Dropout|0.4|
|Dense + Sigmoid|1 unit (binary output)|

## 

## 📊 Results

Evaluated on the HAM10000 test set:

* **Test Accuracy: 72.2%**

||Predicted Benign|Predicted Malignant|
|-|-|-|
|**Actual Benign**|1,110|502|
|**Actual Malignant**|55|336|

The model shows strong recall on the malignant class (\~86%), reflecting the effect of class weighting during training, though the false-positive rate on benign cases suggests room for improvement (e.g., more data, augmentation, deeper architectures).

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pillow matplotlib tensorflow scikit-learn joblib pandas
```

### Dataset

Download the HAM10000 dataset from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) or Keggle and place `HAM10000\_metadata.csv` and the image folders in the expected project directory. Alternatively, the script includes a synthetic lesion-image generator for a no-download demo run.

### Run

```bash
python new\_cancer.py
```

The script will train the model (or load a saved one), evaluate it on the test set, and offer an interactive prompt to check your own skin photo:

```
Would you like to upload a skin photo to check? (y/n): y
Prediction: Malignant (confidence: 80.8%)
```

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **ML Fallback:** scikit-learn (Random Forest)
* **Data Handling:** NumPy, Pandas
* **Image Processing:** Pillow (PIL)
* **Visualization:** Matplotlib
* **Interface:** Tkinter (file-picker for predictions)

## 📁 Project Structure

```
├── new\_cancer.py              # Main script: data loading, model, training, prediction
├── skin\_cancer\_results.png    # Confusion matrix (generated after training)
├── training\_curves.png        # Accuracy/loss curves (generated after training)
└── README.md
```

## 🔮 Future Scope

* Add Precision, Recall, F1-score, and ROC-AUC evaluation
* Data augmentation and transfer learning (ResNet, EfficientNet)
* Hyperparameter tuning and cross-validation
* Extend to the full 7-class lesion taxonomy
* Clinical validation with dermatology professionals before any real-world use

## 📚 Dataset Citation

Tschandl, P., Rosendahl, C. \& Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Sci Data* 5, 180161 (2018). [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

## 📄 License

This project is for academic/educational purposes.

\---

*Developed during an internship in AI \& Machine Learning in Computational Biology at BioTechTrek.*


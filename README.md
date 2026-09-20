# Diabetic-Retinopathy-Diagnosis-using-DenseNet121
This deep learning project detects Diabetic Retinopathy (DR) from retinal fundus images using a fine-tuned DenseNet121 model pretrained on ImageNet. It includes an interactive Streamlit web application where users can upload retinal scans to instantly get predicted classes alongside their confidence scores.

## 📌 Project Overview

Diabetic Retinopathy is a diabetes-related eye disease that can damage the retina and may lead to vision loss. Detecting signs of the disease from retinal fundus photographs is an image-classification problem that can be approached using deep learning.

This project implements a binary image classifier with two classes:

- DR — Diabetic Retinopathy
- No_DR — No Diabetic Retinopathy

The trained model is based on DenseNet121, a convolutional neural network architecture known for efficient feature reuse through dense connections.

The project has two main components:

### 1. Model development and evaluation

- Implemented in the Jupyter/Google Colab notebook.
- Includes dataset discovery, exploratory data analysis, image preprocessing, feature analysis, transfer learning, fine-tuning, evaluation, and error analysis.

### 2. Prediction web application

- Implemented with Streamlit.
- Loads the trained Keras model.
- Accepts JPG, JPEG, and PNG retinal images.
- Displays the uploaded image, predicted class, confidence, and class probabilities.

## ✨ Features

### Machine Learning / Deep Learning

- DenseNet121 transfer learning
- ImageNet-pretrained convolutional backbone
- Input resolution of 224 × 224 × 3
- Data augmentation
- Class-weight handling
- Frozen-backbone training
- Fine-tuning of the DenseNet121 backbone
- Early stopping
- Learning-rate reduction
- Best-model checkpointing
- Binary classification using softmax output
- Test-set evaluation
- Classification report
- Confusion matrix
- Normalized confusion matrix
- ROC curve and ROC-AUC
- Prediction error analysis
- Single-image prediction

### Streamlit Application

- Simple browser-based interface
- Retina image upload
- JPG/JPEG/PNG support
- Uploaded-image preview
- DR / No_DR prediction
- Prediction confidence
- Per-class probabilities
- Probability progress bars
- Cached model loading for better application performance
- Medical-use disclaimer

## 🧠 Model Architecture

The project uses DenseNet121 as the transfer-learning backbone.

### Architecture
```
Input Image
   │
   ▼
224 × 224 × 3
   │
   ▼
DenseNet121
(ImageNet pretrained)
   │
   ▼
Global Average Pooling
   │
   ▼
Dense Layer
256 units + ReLU
   │
   ▼
Batch Normalization
   │
   ▼
Dropout
40%
   │
   ▼
Dense Layer
64 units + ReLU
   │
   ▼
Dropout
25%
   │
   ▼
Output Layer
2 units + Softmax
   │
   ├── DR
   └── No_DR
```
The model contains an internal preprocessing layer based on DenseNet121's preprocess_input function. The Streamlit application therefore supplies the same preprocessing function as a custom object when loading the saved Keras model.

## 🔬 Training Strategy

The model was trained in two major stages.

### Stage 1 — Transfer Learning

The DenseNet121 backbone was initially frozen.

The newly added classification layers were trained using:

- Optimizer: Adam
- Learning rate: 1e-3
- Loss: categorical cross-entropy
- Metric: accuracy
- Maximum epochs: 25

### Stage 2 — Fine-Tuning

After the initial transfer-learning stage, the DenseNet121 backbone was made trainable.

Approximately the later portion of the backbone was fine-tuned while keeping earlier layers frozen. Batch Normalization layers were kept frozen during fine-tuning.

Fine-tuning used:

- Optimizer: Adam
- Learning rate: 1e-5
- Maximum epochs: 20
- Loss: categorical cross-entropy
- Metric: accuracy

### Training Callbacks

The notebook uses:

- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau

The best model is selected based on validation accuracy.

## 🖼️ Data Augmentation

The training pipeline applies image augmentation to improve generalization.

The following transformations are used:

- Rotation: up to 15°
- Width shift: 10%
- Height shift: 10%
- Zoom: 15%
- Shear: 5%
- Brightness range: 0.85–1.15
- Horizontal flip
- Vertical flip
- Nearest-neighbor fill mode

Validation and test images are not augmented.

## 📊 Dataset

The notebook downloads the dataset from Kaggle:

*Diagnosis of Diabetic Retinopathy*

Dataset source:

https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy

The dataset used by the notebook contains the following split:

Here is the text transcribed from the image:

| Split | Number of Images |
| :--- | :--- |
| Training | 2,076 |
| Validation | 531 |
| Testing | 231 |
| **Total** | **2,838** |

The test set contains:

| Class | Test Images |
| :--- | :--- |
| DR | 113 |
| No_DR | 118 |
| **Total** | 231 |

The dataset itself is not included in this GitHub repository. Users should obtain it directly from the original dataset source and follow its terms of use.

## 📈 Model Performance

The following results were produced by the included notebook during the documented training run.

### Final Training / Validation Results

| Metric | Result |
| :--- | :--- |
| Final training accuracy | 97.06% |
| Final validation accuracy | 96.61% |

### Test Set

| Metric | Result |
| :--- | :--- |
| Test accuracy | 98.27% |
| Test loss | 0.0762 |
| ROC-AUC | 0.9951 |

### Per-class metrics

| Class | Precision | Recall | F1 Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| DR | 0.9910[cite: 8] | 0.9735[cite: 8] | 0.9821[cite: 8] | 113[cite: 8] |
| No_DR | 0.9750[cite: 8] | 0.9915[cite: 8] | 0.9832[cite: 8] | 118[cite: 8] |
| **Overall Accuracy** | | | **0.9827**[cite: 8] | **231**[cite: 8] |

There were 4 incorrect predictions in the test set during this run.

## 🔍 Exploratory Data Analysis

The notebook includes several exploratory and statistical analysis steps before model training.

These include:

- Class distribution analysis
- Sample image visualization
- Grayscale image feature extraction
- Mean intensity
- Standard deviation of intensity
- Minimum/maximum intensity
- Median intensity
- Dark-pixel ratio
- Bright-pixel ratio
- Edge strength
- Horizontal variance
- Vertical variance
- Histograms
- Box plots
- Scatter plots
- Pair plots
- Feature scaling
- ANOVA/F-score based feature selection
- Class-weight calculation

These analyses are included primarily for understanding the dataset and supporting the machine-learning workflow. The final image classifier itself relies on DenseNet121 learned visual representations rather than using the engineered statistical features as its input.

## 🧪 Evaluation

The notebook evaluates the trained model using multiple methods.

### Classification Report

Precision, recall, F1-score, and support are calculated separately for both classes.

### Confusion Matrix

A confusion matrix is generated to show:

- Correct DR predictions
- Incorrect DR predictions
- Correct No_DR predictions
- Incorrect No_DR predictions

### Normalized Confusion Matrix

The confusion matrix is also normalized by actual class to make class-wise performance easier to interpret.

### ROC Curve and AUC

For the binary classification task, the notebook calculates the ROC curve and ROC-AUC.

### Error Analysis

Incorrectly classified test images are collected and visualized along with:

- Actual class
- Predicted class
- Prediction confidence

## 🌐 Streamlit Web Application

The file app.py provides a lightweight interface for using the trained model.

### Application workflow
```
Start Streamlit App
        │
        ▼
Load diabetic_retinopathy_model.keras
        │
        ▼
Upload Retina Image
        │
        ▼
Convert to RGB
        │
        ▼
Resize to 224 × 224
        │
        ▼
Model Prediction
        │
        ▼
Get Class Probabilities
        │
        ▼
Display:
  • Prediction
  • Confidence
  • DR probability
  • No_DR probability
```
## 📁 Project Structure
```
Diabetic-Retinopathy-Diagnosis/
│
├── app.py
│
├── Diabetic_Retinopathy_Diagnosis_using_DenseNet121.ipynb
│
├── diabetic_retinopathy_model.keras
│
├── requirements.txt
│
└── README.md
```
### File descriptions
| File | Description |
| :--- | :--- |
| app.py | Streamlit application for making predictions |
| Diabetic_Retinopathy_Diagnosis_using_DenseNet121.ipynb | Complete model development, training, evaluation, and analysis notebook |
| diabetic_retinopathy_model.keras | Trained Keras model used by the Streamlit application |
| requirements.txt | Python dependencies required by the application |
| README.md | Project documentation |

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- DenseNet121
- NumPy
- Pandas
- Pillow
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Google Colab
- Kaggle Dataset

## 📦 Requirements

The Streamlit application requires:

- streamlit
- tensorflow
- pillow
- numpy

The complete notebook additionally uses packages such as:

- pandas
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- kagglehub

A Python version compatible with the installed TensorFlow release should be used.

## 🖥️ How to Use the Application

(1) Start the Streamlit application.

(2) Open the URL provided by Streamlit.

(3) Click Choose a retina image.

(4) Select a retinal fundus image in JPG, JPEG, or PNG format.

(5) The application displays the uploaded image.

(6) The model analyzes the image.

(7) The application displays:

- Predicted class
- rediction confidence
- DR probability
- No_DR probability

Example output:

Prediction: DR

Confidence: 97.35%

Class probabilities:
DR: 97.35%
No_DR: 2.65%

The numbers above are only an example of the interface format and are not a guaranteed prediction.

## 📓 Running the Training Notebook

The notebook was designed for a Google Colab-style workflow.

### General workflow
```
Kaggle Dataset
      │
      ▼
Dataset Download
      │
      ▼
Dataset Discovery
      │
      ▼
EDA & Feature Analysis
      │
      ▼
Data Augmentation
      │
      ▼
DenseNet121 Transfer Learning
      │
      ▼
Initial Training
      │
      ▼
Fine-Tuning
      │
      ▼
Model Evaluation
      │
      ▼
Error Analysis
      │
      ▼
Save Trained Model
```
### Dataset setup

The notebook downloads the Kaggle dataset using the Kaggle API.

You will need a Kaggle account and API credentials if you want to reproduce the notebook's original dataset-download workflow.

The notebook originally uses:

*kaggle datasets download -d pkdarabi/diagnosis-of-diabetic-retinopathy*

After downloading, the dataset is extracted and automatically searched for compatible train, valid, and test directories.

## 🔐 Model Loading

The Streamlit application loads the saved model with:

custom_objects = {
    "preprocess_input": preprocess_input
}

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects=custom_objects
)

The custom_objects argument is important because the saved model contains a preprocessing Lambda layer referencing DenseNet121's preprocess_input function.

Without supplying the preprocessing function when loading the model, model deserialization may fail.

## ⚙️ Prediction Details

For each uploaded image, the application:

(1) Converts the image to RGB.

(2) Resizes it to 224 × 224.

(3) Converts the image into a NumPy array.

(4) Adds a batch dimension.

(5) Sends it to the trained model.

(6) Obtains softmax probabilities.

(7) Selects the class with the highest probability.

The application uses:

IMG_SIZE = 224

and:

CLASS_NAMES = ["DR", "No_DR"]

## 🧾 Reproducibility

The notebook sets:

SEED = 42

and applies this seed to Python's random module, NumPy, and TensorFlow.

Other important configuration values include:

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
FINE_TUNE_EPOCHS = 20
BACKBONE_NAME = "DenseNet121"

Exact reproduction of neural-network training results may still vary depending on:

- TensorFlow version
- CUDA/cuDNN version
- GPU/CPU environment
- Dataset version
- Hardware
- Randomness in the training environment
- Library versions

## 📌 Important Limitations

This project has several important limitations.

### 1. Binary classification

The model predicts only:

DR
No_DR

It does not classify different stages or severity levels of diabetic retinopathy.

### 2. Dataset limitations

The reported performance is based on the dataset and test split used in the notebook. Performance on images from other sources may differ.

### 3. Domain shift

Differences in:

- Camera hardware
- Image resolution
- Lighting
- Image quality
- Patient population
- Clinical environment
- Image acquisition protocol

may affect predictions.

### 4. No clinical validation

The model has not been presented here as clinically validated or approved for medical diagnosis.

### 5. Confidence is not certainty

A high softmax probability does not mean that the prediction is medically certain or clinically reliable.

## 🔮 Possible Future Improvements

Potential improvements include:

- Multi-class diabetic retinopathy severity classification
- Larger and more diverse datasets
- External validation on an independent dataset
- Cross-validation
- Explainable AI using Grad-CAM
- Lesion localization
- Image-quality detection
- Better retinal image preprocessing
- Fundus-image cropping and optic-disc handling
- Hyperparameter optimization
- Comparison with EfficientNet, ResNet, ConvNeXt, or other architectures
- Ensemble models
- Calibration of prediction probabilities
- More comprehensive clinical evaluation
- REST API deployment
- Docker containerization
- Cloud deployment
- Automated model monitoring
- Improved Streamlit UI
- Prediction history
- Exportable prediction reports

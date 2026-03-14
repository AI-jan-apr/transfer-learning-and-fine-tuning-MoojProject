[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/DtxdB3_i)

# Transfer Learning with EfficientNet on Food11

This project applies **transfer learning** using **EfficientNetB0** to classify food images from the **Food11 dataset**.

The repository contains **two notebook versions**.  
Both implement the same idea but with different levels of complexity.

---

# Dataset — Food11

The dataset contains images of food divided into **11 classes**:

- Bread  
- Dairy product  
- Dessert  
- Egg  
- Fried food  
- Meat  
- Noodles-Pasta  
- Rice  
- Seafood  
- Soup  
- Vegetable-Fruit  

Each image is resized to **224 × 224** to match the input size of EfficientNet.

---

# Notebook Versions

## 1. Simple Version  
`efficientnet_transfer_learning_simple.ipynb`

Made for learning and understanding

The code is clean and easy to follow


- Loading the dataset
- Basic preprocessing
- Data augmentation (flip, rotate, zoom)
- EfficientNetB0 backbone with a small classification head
- Running the four experiments sequentially
- Evaluation on the test set
- Confusion matrix
- Classification report
- Basic MLflow logging (accuracy and loss)

### What it does not include
- EarlyStopping or ModelCheckpoint
- Dataset optimization (AUTOTUNE / prefetch)
- Dropout layer
- Separate training curves per experiment
- Full MLflow experiment tracking
- Layer-wise learning rate decay

---

## 2. Advanced Version  
`Transfer_Learning_EfficientNet.ipynb`

### What it includes
- Everything in the simple version
- EarlyStopping and ModelCheckpoint callbacks
- Dataset optimization with AUTOTUNE
- Dropout in the classification head
- MLflow + DagsHub integration
- Separate plots for each experiment
- Layer-wise learning rate decay
- Final comparison chart 

---

# Experiments (Advanced Version)

## Experiment 1 — Feature Extraction
The EfficientNet backbone is **fully frozen** and only the classification head is trained.

Test Accuracy: **88.77%**  
Test Loss: **0.3512**

---

## Experiment 2 — Fine-Tuning
The **last 20 layers** of EfficientNet are unfrozen and trained with a small learning rate.

Learning rate: `1e-5`

Test Accuracy: **90.21%**  
Test Loss: **0.3110**

---

## Experiment 3 — Gradual Unfreezing
The model is unfrozen **in stages**.

Phase 1: last 40 layers  
Phase 2: last 10 layers  

Test Accuracy: **90.99%**  
Test Loss: **0.3005**

---

## Experiment 4 — Layer-wise Learning Rate Decay
Different parts of the network use different learning rates.

Earlier layers → smaller LR  
Later layers → larger LR

Test Accuracy: **91.29%**  
Test Loss: **0.2947**

This experiment achieved the **best result**.

---

# Simple Version Results

In the simple notebook all experiments were run sequentially on the same model.

Final results:

Test Accuracy: **91.84%**  
Test Loss: **0.3169**

### Per-class performance

| Class | Precision | Recall | F1 |
|------|------|------|------|
| Bread | 0.90 | 0.87 | 0.88 |
| Dairy product | 0.92 | 0.75 | 0.83 |
| Dessert | 0.88 | 0.89 | 0.89 |
| Egg | 0.89 | 0.91 | 0.90 |
| Fried food | 0.89 | 0.89 | 0.89 |
| Meat | 0.93 | 0.93 | 0.93 |
| Noodles-Pasta | 0.99 | 0.97 | 0.98 |
| Rice | 0.91 | 0.95 | 0.93 |
| Seafood | 0.93 | 0.93 | 0.93 |
| Soup | 0.96 | 0.99 | 0.97 |
| Vegetable-Fruit | 0.94 | 0.98 | 0.96 |

---

## Tools Used

- Python  
- TensorFlow / Keras  
- EfficientNetB0 (pretrained on ImageNet)  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- MLflow  
- DagsHub  

---

## Project Structure

```
transfer-learning-and-fine-tuning-MoojProject/
│
├── notebooks/
│   ├── efficientnet_transfer_learning_simple.ipynb   
│   └── Transfer_Learning_EfficientNet.ipynb          
│
├── models/
│   ├── feature_extraction_best.keras
│   ├── fine_tuned_best.keras
│   ├── exp3_gradual_unfreezing.keras
│   └── exp4_layerwise_lr_decay.keras
│
├── results/
│   ├── comparison.csv
│   ├── comparison_chart.png
│   ├── feature_extraction_curves.png
│   ├── fine_tuning_curves.png
│   ├── gradual_unfreezing_curves.png
│   ├── layerwise_lr_curves.png
│   └── final_summary.json
│
└── README.md
```
## 🔗 Helpful Links

- 📚 EfficientNet models in Keras:  
  https://keras.io/api/applications/efficientnet/

- 🎓 Transfer Learning guide (Keras):  
  https://keras.io/guides/transfer_learning/

- 📦 MLflow for experiment tracking:  
  https://www.mlflow.org/docs/latest/index.html

- ☁️ DVC + DagsHub integration:  
  https://dagshub.com/docs/integrations/dvc/

- 🧑‍🍳 How to freeze/unfreeze layers in Keras:  
  https://keras.io/getting_started/faq/#how-can-i-freeze-layers-in-a-model

- 📈 Using callbacks in Keras (e.g. EarlyStopping, ReduceLROnPlateau):  
  https://keras.io/api/callbacks/




<!-- ## 🧠 Task Overview

You will apply **Transfer Learning** using **EfficientNet** models with two approaches:  
1. **Feature Extraction**  
2. **Fine-tuning**

⚠️ This task **must be completed in Google Colab or a cloud-based environment**. Training deep models like EfficientNet on local machines without GPU/TPU is highly inefficient and may lead to failed or incomplete experiments.



## 📁 Dataset

Dataset is already downloaded and loaded in the notebook. Preprocess as needed for training.



## 🧪 Experiments

### 1️⃣ Feature Extraction  
- freeze all base layers  
- train only the classification head  

### 2️⃣ Fine-tuning  
- unfreeze last layers  
- retrain full or partial base  

You can enhance fine-tuning with these techniques:

- **Unfreeze only last *n* layers**  
  gradually increase trainable layers instead of full base model

- **Gradual unfreezing**  
  unfreeze layers one block at a time across training epochs

- **Layer-wise learning rate decay**  
  assign smaller LR to earlier layers and higher LR to deeper layers

For each:
- document model version  
- include training/validation metrics  
- write your analysis



## 🧬 Bonus (Optional)

- use **DagsHub** to upload and manage dataset in a cloud bucket  
- track all runs using **MLflow**:
  - versioned experiments  
  - parameters, metrics, artifacts  

## 📝 README Must Include:

- experiment summary  
- plots for metrics  
- observations on:
  - feature extract vs fine-tune  
  - generalization, convergence, overfitting  -->

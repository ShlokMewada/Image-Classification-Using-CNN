# Image-Classification-Using-CNN

# Intel Image Classification using CNN (TensorFlow/Keras)

A deep learning project built using Convolutional Neural Networks (CNN) to classify natural scenes into 6 categories: **mountain, street, buildings, sea, forest, glacier**.

Achieved **~89.3% test accuracy** using a custom CNN trained from scratch with real-time data augmentation, early stopping, and learning rate scheduling.

---

## Dataset

- Dataset: [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

---

## Project Highlights

- Built custom CNN model (3 Conv blocks, BatchNorm, Dropout)
- Used **ImageDataGenerator** for augmentation (train only)
- Used **EarlyStopping** and **ReduceLROnPlateau**
- Evaluated using confusion matrix and classification report
- Tested on custom unseen images
- Achieved **Test Accuracy: 89.3%**

---

## Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn

---

## Model Performance

| Metric        | Score    |
|---------------|----------|
| **Train Acc** | 93.2%    |
| **Val Acc**   | 88.9%    |
| **Test Acc**  | 89.3%    |
| **Test Loss** | 0.3389   |

## Class-wise F1 Scores

| Class     | F1-Score |
|-----------|----------|
| buildings | 0.88     |
| forest    | 0.97     |
| glacier   | 0.85     |
| mountain  | 0.85     |
| sea       | 0.90     |
| street    | 0.91     |

---



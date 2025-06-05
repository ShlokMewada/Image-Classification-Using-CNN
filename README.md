# Image-Classification-Using-CNN

# Intel Image Classification using CNN (TensorFlow/Keras)

A deep learning project built using Convolutional Neural Networks (CNN) to classify natural scenes into 6 categories: **mountain, street, buildings, sea, forest, glacier**.

Achieved **~88.5% test accuracy** using a custom CNN trained from scratch with real-time data augmentation, early stopping, and learning rate scheduling.

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
- Achieved **Test Accuracy: 88.5%**

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
| **Train Acc** | 93.1%    |
| **Val Acc**   | 88.6%    |
| **Test Acc**  | 88.5%    |
| **Test Loss** | 0.3396   |

## Class-wise F1 Scores

| Class     | F1-Score |
|-----------|----------|
| buildings | 0.88     |
| forest    | 0.97     |
| glacier   | 0.84     |
| mountain  | 0.84     |
| sea       | 0.90     |
| street    | 0.90     |

---

#### image-classification-project.ipynb contains project code.



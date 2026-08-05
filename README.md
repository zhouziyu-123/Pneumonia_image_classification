# Pneumonia_image_classification
A deep learning project investigating the effectiveness of **transfer learning** for pneumonia classification from chest X-ray images.

This project compares multiple MobileNetV2-based training strategies under different training dataset sizes to evaluate how transfer learning improves performance on small medical imaging datasets.

## Overview
Medical image classification often suffers from limited labeled data. This project evaluates whether transfer learning can improve classification performance when only a small number of training images are available.

Three different training strategies were implemented using **MobileNetV2** as the backbone:

- **Model A** – Training from scratch
- **Model B** – Frozen pretrained MobileNetV2 (Transfer Learning)
- **Model C** – Partial fine-tuning of MobileNetV2

Each strategy was evaluated using **10%, 25%, 50%, and 100%** of the training dataset to study the relationship between dataset size and model performance.:contentReference[oaicite:1]{index=1}

## Dataset
Dataset:

**Chest X-ray Pneumonia Dataset**

Source:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Dataset characteristics

- 5,856 pediatric chest X-ray images
- Binary classification
    - Normal
    - Pneumonia
- Multi-class classification
    - Normal
    - Bacterial Pneumonia
    - Viral Pneumonia

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── .gitignore
│
├── notebooks/
│   ├── main.ipynb
│   ├── Model_A.ipynb
│   ├── Model_B.ipynb
│   ├── Model_C.ipynb
│   ├── Model_A_3Classifier.ipynb
│   ├── Model_B_3Classifier.ipynb
│   ├── Model_C_3Classifier.ipynb
│   └── Model_C_Improve_3Classifier.ipynb
│
├── models/
│   ├── best_model.keras
│   ├── *.h5
│
└── chest_xray/
```

## Model Architectures

### Model A

Training MobileNetV2 from scratch.

- Random weight initialization
- Entire network trained end-to-end

### Model B

Transfer learning with a frozen backbone.

- ImageNet pretrained MobileNetV2
- Frozen feature extractor
- Custom classification head

### Model C

Partial fine-tuning.

- Initialized from Model B
- Last MobileNetV2 block unfrozen
- Smaller learning rate
- Improved classification head tested

All models were evaluated for both binary and multi-class classification tasks.:contentReference[oaicite:3]{index=3}


## Training Strategy

Optimizer

- Adam

Learning rate

- Scratch: 1e-3
- Frozen Backbone: 1e-4
- Fine-tuning: 1e-5

Other settings

- Batch size: 32
- Maximum epochs: 50
- Early stopping
- Best model selected using validation accuracy:contentReference[oaicite:4]{index=4}

## Results

Major findings include

- Training from scratch performs poorly on small datasets.
- Transfer learning achieves over 90% accuracy even with only 10% of the training data.
- Fine-tuning improves performance only when combined with an improved classification head.
- Transfer learning consistently outperforms training from scratch for medical image classification.:contentReference[oaicite:6]{index=6}

## Technologies

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Skills Demonstrated

This project demonstrates experience in

- Medical image analysis
- Deep learning
- Transfer learning
- Binary classification
- Multi-class classification
- CNN model development
- Hyperparameter tuning
- Performance evaluation
- Experimental comparison
- Scientific data analysis


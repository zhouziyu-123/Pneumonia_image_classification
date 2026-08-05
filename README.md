## Pneumonia_image_classification
A deep learning project investigating the effectiveness of **transfer learning** for pneumonia classification from chest X-ray images.

This project compares multiple MobileNetV2-based training strategies under different training dataset sizes to evaluate how transfer learning improves performance on small medical imaging datasets.

# Overview
Medical image classification often suffers from limited labeled data. This project evaluates whether transfer learning can improve classification performance when only a small number of training images are available.

Three different training strategies were implemented using **MobileNetV2** as the backbone:

- **Model A** – Training from scratch
- **Model B** – Frozen pretrained MobileNetV2 (Transfer Learning)
- **Model C** – Partial fine-tuning of MobileNetV2

Each strategy was evaluated using **10%, 25%, 50%, and 100%** of the training dataset to study the relationship between dataset size and model performance.:contentReference[oaicite:1]{index=1}

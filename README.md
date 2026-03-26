# Homework 2: Vision Transformer (Variant A)
**Author:** Adam Al Subhi  
**Institution:** HSE University  

## Overview
This repository contains the implementation of a custom Vision Transformer (ViT) from scratch using PyTorch. The model is built to classify images from the CIFAR-10 dataset (32x32 resolution). This project is submitted as part of Homework 2, focusing on tokenized representations and self-attention architectures in Computer Vision.

## Repository Structure
* `HW2_Vision_Transformer.ipynb`: The complete Google Colab notebook containing data loading, the ViT architecture (Patch Embedding, Transformer Blocks, Classification Head), and the training/evaluation loop.
* `report.md`: A concise analysis of the model's performance, attention cost, and a detailed error analysis based on different configurations.
* `confusion_matrix.png`: A heatmap visualization of the model's predictions to highlight common misclassifications.

## Model Architecture Details
The architecture strictly follows a token-based scheme:
* **Patch Embedding:** Uses `nn.Conv2d` to extract non-overlapping patches and flatten them into a sequence.
* **Tokens:** Incorporates a learnable `[CLS]` token and standard Additive Positional Embeddings.
* **Transformer Block:** Implements standard Layer Normalization, Multi-Head Self-Attention, and an MLP layer with GELU activation.

## How to Run
The code was developed and tested on Google Colab. To reproduce the results:
1. Open the `.ipynb` file in Google Colab.
2. Ensure the hardware accelerator is set to GPU (`Runtime` -> `Change runtime type` -> `T4 GPU`) for faster training.
3. Run all cells sequentially. The CIFAR-10 dataset will be downloaded automatically via `torchvision`.

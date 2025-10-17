# 🧠 Domain Adaptation for Robust Object Detection (DA-YOLO)

## 🎯 Introduction

This project implements and evaluates **Domain Adaptation (DA)** techniques to enhance the robustness of state-of-the-art object detection models (**YOLOv8**) when applied to new, unseen target domains (e.g., changing weather conditions, different sensor data).

The goal is to transfer knowledge from a labeled source domain (**Source Domain - S**) to an unlabeled or sparsely labeled target domain (**Target Domain - T**), ensuring consistent performance without extensive re-labeling.

---

## ✨ Key Techniques

| Model/Technique | Description | File |
|-----------------|--------------|------|
| **Domain-Adversarial Neural Network (DANN)** | Used to learn domain-invariant features by adding a Domain Discriminator and a Gradient Reversal Layer (GRL). | `train_yolov8s_DA.ipynb` |


---

## 🛠️ Setup Instructions

### 1. Prerequisites

- Python 3.8+
- PyTorch 

---

### 2. Install environment.
- Using `conda` to create virtual environment.

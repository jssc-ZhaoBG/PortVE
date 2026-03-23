# PortVE
# 📌 Lightweight Visual Enhancement for Reliable Vision Systems in Adverse Port Weather

## Introduction
This repository provides the official implementation of the paper:

**Lightweight Visual Enhancement for Reliable Vision Systems in Adverse Port Weather**

In this work, we propose a lightweight visual enhancement framework designed to improve the robustness and reliability of vision-based systems operating under adverse port weather conditions such as haze, fog, and rain. The method enhances degraded images while maintaining efficiency for deployment on edge or embedded devices.

---

## Reference Code
This project is built upon and inspired by the following repositories:

[ConvIR](https://github.com/c-yn/ConvIR)

We sincerely thank the authors for their excellent work.

---

## Installation

The project is built with the following environment:

- Python 3.8  
- PyTorch 1.8.0  
- CUDA 11.1

pip install tensorboard einops scikit-image pytorch_msssim opencv-python
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..

## Training and Evaluation

Training
python main.py --mode train --data_dir your_path

Test
python main.py --mode test --data_dir your_path --test_model path_to_model










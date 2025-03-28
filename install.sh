#!/bin/bash

echo "🔧 Installation des paquets depuis requirements.txt (hors PyTorch)..."
pip install -r requirements.txt

echo "⚡ Installation de PyTorch + torchvision avec support CUDA 12.6..."
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126

echo "✅ Installation terminée."

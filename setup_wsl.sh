#!/usr/bin/env bash
set -e

echo ""
echo "=============================================="
echo "  WSL2 TensorFlow GPU Setup"
echo "=============================================="

echo ""
echo "--- Step 1: Create venv ---"
python3 -m venv ~/vf-venv
source ~/vf-venv/bin/activate
python -m pip install --upgrade pip wheel setuptools -q
pip --version

echo ""
echo "--- Step 2: Install PyTorch (CUDA 12.8) ---"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q
python -c "import torch; print('torch    :', torch.__version__); print('CUDA     :', torch.cuda.is_available()); avail=torch.cuda.is_available(); print('GPU      :', torch.cuda.get_device_name(0) if avail else 'N/A')"

echo ""
echo "--- Step 3: Install TensorFlow GPU ---"
pip install "tensorflow[and-cuda]" -q
python -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print('tf       :', tf.__version__); print('GPU count:', len(gpus)); [print('  ', g.name) for g in gpus]"

echo ""
echo "--- Step 4: Install project requirements ---"
pip install numpy pandas scikit-learn arch xgboost matplotlib seaborn scipy statsmodels yfinance joblib optuna -q
echo "Requirements installed."

echo ""
echo "=============================================="
echo "  WSL2 Setup Complete!"
echo "  Activate: source ~/vf-venv/bin/activate"
echo "=============================================="
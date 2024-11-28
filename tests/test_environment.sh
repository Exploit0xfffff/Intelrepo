#!/bin/bash

# Test script for verifying the Deep Actions Experimental environment

echo "=== Testing Deep Actions Experimental Setup ==="

# Check Python version
echo "Checking Python version..."
python3 --version

# Create fresh test environment
echo "Creating test environment..."
TEST_DIR="/tmp/deep-actions-test"
rm -rf $TEST_DIR
mkdir -p $TEST_DIR
cd $TEST_DIR

# Create virtual environment
echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Clone repository
echo "Cloning repository..."
git clone https://github.com/Exploit0xfffff/Deep-Actions-Experimental .

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install system dependencies
echo "Installing system dependencies..."
if ! command -v apt-get &> /dev/null; then
    echo "WARNING: apt-get not found. Please install GTK dependencies manually:"
    echo "python3-gi python3-gi-cairo gir1.2-gtk-3.0"
else
    sudo apt-get install -y python3-gi python3-gi-cairo gir1.2-gtk-3.0
fi

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if [ $? -eq 0 ]; then
    echo "PyTorch CUDA check: OK"
else
    echo "PyTorch CUDA check: Failed"
fi

# Test model initialization
echo "Testing model initialization..."
python3 -c "
from main.model_factory import ModelFactory
print('Testing Faster R-CNN initialization...')
model, device = ModelFactory.create_model('fasterrcnn')
print('Faster R-CNN initialization: OK')
print('Testing YOLO initialization...')
model, device = ModelFactory.create_model('yolo')
print('YOLO initialization: OK')
"

echo "=== Environment Test Complete ==="
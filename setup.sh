#!/bin/bash

echo "Setting up Passenger Monitoring System..."

mkdir -p ~/passenger-monitoring
cd ~/passenger-monitoring

echo "Creating virtual environment..."
python -m venv --system-site-packages venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install tensorflow opencv-python numpy ultralytics pillow psutil

echo "Installing Raspberry Pi specific packages..."
pip install picamera2

echo "Creating directory structure..."
mkdir -p models output

echo "Downloading YOLOv8 model..."
if [ ! -f "models/yolov8n.pt" ]; then
    wget -O models/yolov8n.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    echo "YOLOv8 model downloaded successfully!"
else
    echo "YOLOv8 model already exists."
fi

echo ""
echo "Note: Violence detection model needs to be trained and converted to TensorFlow format"

chmod +x main.py

echo ""
echo "Setup completed!"
echo ""
echo "To run the system:"
echo "  cd ~/passenger-monitoring"
echo "  source venv/bin/activate"
echo "  python main.py"

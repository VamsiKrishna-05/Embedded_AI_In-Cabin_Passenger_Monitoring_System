# Embedded AI-Based In-Cabin Surveillance System For Fully Autonomous Cabs

A real-time video-based passenger monitoring system with embedded AI-powered violence detection capabilities designed for Raspberry Pi and other edge devices in autonomous vehicles.

## Features

- **Passenger Detection**: Uses YOLOv8 for real-time person detection
- **Violence Detection**: TensorFlow-based deep learning model with fallback motion analysis
- **Alert System**: Automatic alerts when violence is detected
- **Performance Monitoring**: Real-time FPS, CPU, and memory usage tracking
- **Video Recording**: Optional video output of monitoring sessions
- **Multi-Camera Support**: Supports both Raspberry Pi cameras and standard webcams

## System Architecture

```
PassengerMonitoringSystem (Main)
├── PassengerDetector (YOLOv8)
├── ViolenceDetector (TensorFlow)
├── AlertSystem (Logging & Alerts)
└── PerformanceMonitor (Metrics)
```

## Requirements

- Python 3.8+
- OpenCV
- TensorFlow 2.10+
- Ultralytics YOLO
- psutil
- picamera2 (for Raspberry Pi)

## Installation

### Quick Setup (Linux/Raspberry Pi)

```bash
bash setup.sh
```

### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p models output
```

## Configuration

Edit `config.py` to customize:
- Camera resolution (default: 640x480)
- FPS (default: 15)
- Confidence thresholds
- Model paths
- Output directory

### Key Configuration Parameters

```python
SEQUENCE_LENGTH = 20       # Frames for violence detection
FRAME_SIZE = (112, 112)   # Model input size
VIOLENCE_THRESHOLD = 0.7  # Detection confidence
```

## Usage

### Running the System

```bash
source venv/bin/activate
python main.py
```

### Controls

- Press `q` to quit the application
- Output videos are saved in the `output/` directory
- Alerts are logged to `alerts.log`

## Model Requirements

### YOLOv8 (Passenger Detection)
- Automatically downloaded on first run
- Detects people/passengers in real-time

### Violence Detection Model
- Expected input: 20 frames of 112x112 pixels
- Format: TensorFlow H5 model
- Place at: `models/violencemagicvit.h5`
- If not available, system uses motion-based fallback

## Output Files

- `alerts.log` - Alert history with timestamps
- `output/output_YYYYMMDD_HHMMSS.avi` - Recorded video sessions
- Console output with real-time metrics

## Performance Metrics

The system displays:
- **FPS**: Frames per second
- **CPU Usage**: Percentage
- **Passenger Count**: Number of detected passengers
- **Violence Status**: Current threat level with confidence score

## Fallback Detection

If TensorFlow model is unavailable, the system automatically switches to motion-based violence detection using:
- Average motion magnitude
- Peak motion values
- Motion variance analysis

## Raspberry Pi Considerations

- Use Pi Camera v2 for best results
- System prioritizes Pi Camera over USB webcams
- Recommend GPU acceleration for YOLOv8
- Monitor temperature during extended use

## Troubleshooting

### Camera Not Found
- Check camera connection
- Verify camera index in config.py
- Try different camera indices (0-3)

### Model Loading Issues
- Ensure models directory exists
- Check TensorFlow version compatibility
- Verify model file formats

### Performance Issues
- Reduce FPS setting
- Lower camera resolution
- Disable video saving if not needed

## Project Structure

```
passenger-monitoring-system/
├── main.py                    # Main application
├── config.py                  # Configuration settings
├── passenger_detector.py       # YOLOv8 detection
├── violence_detector.py        # Violence detection logic
├── alert_system.py            # Alert handling
├── performance_monitor.py      # Metrics tracking
├── requirements.txt           # Python dependencies
├── setup.sh                   # Automated setup script
└── README.md                  # This file
```

## License

MIT License - Feel free to use and modify for your projects

## Author

Vamsi

## Support

For issues and questions, please refer to the project documentation or create an issue in the repository.

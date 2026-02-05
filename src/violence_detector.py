import cv2
import numpy as np
import time
import os
from config import config

class ViolenceDetector:
    def __init__(self):
        self.tensorflow_available = False
        self.model = None

        try:
            import tensorflow as tf
            self.tf = tf
            if os.path.exists(config.VIOLENCE_MODEL_PATH):
                self.model = tf.keras.models.load_model(config.VIOLENCE_MODEL_PATH)
                self.tensorflow_available = True
                print("TensorFlow violence model loaded successfully!")
                print(f"Model expects: {config.SEQUENCE_LENGTH} frames of size {config.FRAME_SIZE}")
            else:
                print("Violence model not found. Using fallback detection.")
        except Exception as e:
            print(f"TensorFlow not available or failed to load model: {e}")
            print("Using fallback violence detection with motion analysis.")

        self.frame_buffer = []
        self.buffer_size = config.SEQUENCE_LENGTH

    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, config.FRAME_SIZE)
        frame = frame.astype('float32') / 255.0
        return frame

    def add_frame(self, frame):
        processed_frame = self.preprocess_frame(frame)
        self.frame_buffer.append(processed_frame)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

    def detect_violence_tensorflow(self):
        if len(self.frame_buffer) < self.buffer_size:
            return 0.1, "Calibrating..."

        input_array = np.stack(self.frame_buffer, axis=0)
        input_array = np.expand_dims(input_array, axis=0)

        predictions = self.model.predict(input_array, verbose=0)
        
        # Handle 2-class output: [non_violence_prob, violence_prob]
        if predictions.shape[1] == 2:
            violence_prob = float(predictions[0][1])  # Second class is violence
        else:
            violence_prob = float(predictions[0][0])
        
        status = "Violence Detected" if violence_prob > config.VIOLENCE_THRESHOLD else "No Violence"
        return violence_prob, status

    def detect_violence_fallback(self):
        if len(self.frame_buffer) < 2:
            return 0.1, "Calibrating..."

        motion_values = []
        for i in range(1, len(self.frame_buffer)):
            frame1 = (self.frame_buffer[i - 1] * 255).astype('uint8')
            frame2 = (self.frame_buffer[i] * 255).astype('uint8')
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            motion = np.mean(diff)
            motion_values.append(motion)

        if motion_values:
            avg_motion = np.mean(motion_values)
            max_motion = np.max(motion_values)
            motion_variance = np.var(motion_values) if len(motion_values) > 1 else 0
            violence_prob = min(
                (avg_motion / 30.0) * 0.4 +
                (max_motion / 60.0) * 0.3 +
                (motion_variance / 100.0) * 0.3,
                0.95
            )
            violence_prob = max(violence_prob, 0.1)
        else:
            violence_prob = 0.1

        status = "Violence Detected" if violence_prob > config.VIOLENCE_THRESHOLD else "No Violence"
        return violence_prob, status

    def detect_violence(self):
        if self.tensorflow_available and self.model is not None:
            return self.detect_violence_tensorflow()
        else:
            return self.detect_violence_fallback()

#!/usr/bin/env python3

import cv2
import time
import numpy as np
from datetime import datetime
from config import config
from passenger_detector import PassengerDetector
from violence_detector import ViolenceDetector
from performance_monitor import PerformanceMonitor
from alert_system import AlertSystem

class PassengerMonitoringSystem:
    def __init__(self):
        print("Initializing Passenger Monitoring System...")
        config.download_yolo_model()
        self.passenger_detector = PassengerDetector()
        self.violence_detector = ViolenceDetector()
        self.performance_monitor = PerformanceMonitor()
        self.alert_system = AlertSystem()
        self.cap = None
        self.video_writer = None
        self.passenger_count = 0
        self.violence_status = "No Violence"
        self.violence_probability = 0.0
        self.initialize_camera()
    
    def initialize_camera(self):
        """Initialize camera with better error handling"""
        # Try Pi Camera first
        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            camera_config = self.picam2.create_preview_configuration(
                main={"format": 'XRGB8888', "size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)}
            )
            self.picam2.configure(camera_config)
            self.picam2.start()
            time.sleep(2)
            print("Pi Camera initialized successfully!")
            self.camera_type = "pi"
            return
        except Exception as e:
            print(f"Pi Camera failed: {e}")
            self.picam2 = None
        
        # Try different webcam indices
        for camera_index in [0, 1, 2, 3]:
            try:
                self.cap = cv2.VideoCapture(camera_index)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
                
                # Test if camera works
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    print(f"Webcam at index {camera_index} initialized successfully!")
                    self.camera_type = "webcam"
                    return
                else:
                    self.cap.release()
            except Exception as e:
                if self.cap:
                    self.cap.release()
                print(f"Webcam index {camera_index} failed: {e}")
        
        print("ERROR: No camera found!")
        self.camera_type = None
    
    def initialize_video_writer(self):
        if config.SAVE_VIDEO and self.camera_type is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{config.OUTPUT_DIR}/output_{timestamp}.avi"
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, config.FPS, 
                (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
            )
    
    def draw_detections(self, frame, passengers):
        for i, passenger in enumerate(passengers):
            x, y, w, h = passenger['bbox']
            confidence = passenger['confidence']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Passenger {i+1}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Passengers: {len(passengers)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        color = (0, 0, 255) if self.violence_status == "Violence Detected" else (0, 255, 0)
        cv2.putText(frame, f"Status: {self.violence_status} ({self.violence_probability:.2f})", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        metrics = self.performance_monitor.get_metrics()
        cv2.putText(frame, f"FPS: {metrics['fps']:.1f}", 
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"CPU: {metrics['cpu_usage']:.1f}%", 
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, config.CAMERA_HEIGHT - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def process_frame(self, frame):
        self.performance_monitor.update()
        passengers = self.passenger_detector.detect_passengers(frame)
        self.passenger_count = len(passengers)
        
        if passengers:
            rear_passengers = self.extract_rear_passengers(frame, passengers)
            if rear_passengers:
                largest_passenger = max(rear_passengers, key=lambda p: p[2] * p[3])
                x, y, w, h = largest_passenger
                passenger_roi = frame[y:y+h, x:x+w]
                
                if passenger_roi.size > 0:
                    self.violence_detector.add_frame(passenger_roi)
                    self.violence_probability, self.violence_status = self.violence_detector.detect_violence()
        
        self.alert_system.check_violence_status(self.violence_status, self.violence_probability)
        frame = self.draw_detections(frame, passengers)
        return frame
    
    def extract_rear_passengers(self, frame, passengers):
        height, width = frame.shape[:2]
        rear_region_threshold = height * 0.6
        rear_passengers = []
        for passenger in passengers:
            x, y, w, h = passenger['bbox']
            if y + h > rear_region_threshold:
                rear_passengers.append((x, y, w, h))
        return rear_passengers
    
    def run(self):
        if self.camera_type is None:
            print("Failed to initialize any camera!")
            return
        
        self.initialize_video_writer()
        print("Starting passenger monitoring system...")
        print("Press 'q' to quit")
        
        try:
            while True:
                # Capture frame based on camera type
                if self.camera_type == "pi":
                    frame = self.picam2.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to capture frame from webcam")
                        break
                
                processed_frame = self.process_frame(frame)
                cv2.imshow("Passenger Monitoring", processed_frame)
                
                if self.video_writer is not None:
                    self.video_writer.write(processed_frame)
                
                self.print_status()
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("\nStopping monitoring system...")
        finally:
            self.cleanup()
    
    def print_status(self):
        metrics = self.performance_monitor.get_metrics()
        status_msg = (f"Passengers: {self.passenger_count} | "
                      f"Status: {self.violence_status} | "
                      f"Confidence: {self.violence_probability:.2f} | "
                      f"FPS: {metrics['fps']:.1f}")
        print(status_msg, end='\r')
    
    def cleanup(self):
        if hasattr(self, 'picam2') and self.picam2 is not None:
            self.picam2.stop()
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()
        print("\nMonitoring system stopped.")

if __name__ == "__main__":
    system = PassengerMonitoringSystem()
    system.run()

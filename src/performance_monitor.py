import time
import psutil

class PerformanceMonitor:
    def __init__(self):
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.cpu_usage = 0
        self.memory_usage = 0
        
    def update(self):
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = current_time
            self.cpu_usage = psutil.cpu_percent()
            self.memory_usage = psutil.virtual_memory().percent
    
    def get_metrics(self):
        return {
            'fps': self.fps,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage
        }

"""
Data Collection Module for Hand Gesture Recognition
Captures hand gestures from webcam and saves them for annotation/training.
"""

import cv2
import os
import numpy as np
from pathlib import Path
from datetime import datetime


class HandGestureCollector:
    """Captures and saves hand gesture images from webcam."""
    
    def __init__(self, output_dir='data/raw', image_size=(224, 224)):
        """
        Initialize the gesture collector.
        
        Args:
            output_dir: Directory to save captured images
            image_size: Size of images to capture (width, height)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.capture = None
        self.gesture_name = None
        self.frame_count = 0
        
    def start_capture(self, gesture_name):
        """
        Start capturing a specific gesture.
        
        Args:
            gesture_name: Name of the gesture to capture (e.g., 'palm', 'fist', 'victory')
        """
        self.gesture_name = gesture_name
        gesture_dir = self.output_dir / gesture_name
        gesture_dir.mkdir(parents=True, exist_ok=True)
        
        # Find next available sequence number
        existing = list(gesture_dir.glob('*.jpg'))
        self.frame_count = len(existing)
        
        print(f"Starting capture for gesture: {gesture_name}")
        print(f"Images will be saved to: {gesture_dir}")
        print(f"Starting from frame: {self.frame_count}")
        
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Error: Cannot open webcam")
            return False
        
        # Set camera properties
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        
        self._capture_loop()
        return True
    
    def _capture_loop(self):
        """Main capture loop with keyboard controls."""
        print("\nControls:")
        print("  SPACE  - Capture frame")
        print("  's'    - Save gesture batch")
        print("  'q'    - Quit and save")
        print("\n" + "="*50)
        
        captured_frames = 0
        
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
            
            # Flip frame for selfie-view
            frame = cv2.flip(frame, 1)
            
            # Add instruction overlay
            cv2.putText(frame, f'Gesture: {self.gesture_name}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Captured: {captured_frames}', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, 'SPACE: Capture | s: Save | q: Quit', (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw ROI rectangle
            h, w = frame.shape[:2]
            roi_size = 300
            x1 = (w - roi_size) // 2
            y1 = (h - roi_size) // 2
            x2 = x1 + roi_size
            y2 = y1 + roi_size
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            cv2.imshow('Hand Gesture Capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Spacebar - capture frame
                # Extract ROI (Region of Interest)
                roi = frame[y1:y2, x1:x2]
                roi_resized = cv2.resize(roi, self.image_size)
                
                # Save frame
                gesture_dir = self.output_dir / self.gesture_name
                filename = gesture_dir / f'{self.frame_count:06d}.jpg'
                cv2.imwrite(str(filename), roi_resized)
                
                self.frame_count += 1
                captured_frames += 1
                print(f"Saved: {filename}")
                
            elif key == ord('s'):  # 's' - save batch summary
                print(f"\nBatch saved! Total captured: {captured_frames} frames")
                captured_frames = 0
                
            elif key == ord('q'):  # 'q' - quit
                print(f"\nCapture completed! Total frames: {self.frame_count}")
                break
        
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()
    
    def list_gestures(self):
        """List all captured gesture classes."""
        gestures = [d.name for d in self.output_dir.iterdir() if d.is_dir()]
        return sorted(gestures)
    
    def get_gesture_stats(self):
        """Get statistics about captured gestures."""
        stats = {}
        for gesture_dir in self.output_dir.iterdir():
            if gesture_dir.is_dir():
                image_count = len(list(gesture_dir.glob('*.jpg')))
                stats[gesture_dir.name] = image_count
        return stats


def main():
    """Main function to run the data collector."""
    collector = HandGestureCollector(output_dir='data/raw')
    
    print("="*50)
    print("Hand Gesture Recognition - Data Collector")
    print("="*50)
    
    # Define gestures to capture
    gestures = ['palm', 'fist', 'victory', 'ok', 'thumbs_up']
    
    for gesture in gestures:
        print(f"\n[{gestures.index(gesture)+1}/{len(gestures)}]")
        response = input(f"Ready to capture '{gesture}'? (y/n): ").lower()
        if response == 'y':
            collector.start_capture(gesture)
    
    # Print statistics
    print("\n" + "="*50)
    print("Capture Statistics:")
    print("="*50)
    stats = collector.get_gesture_stats()
    for gesture, count in sorted(stats.items()):
        print(f"{gesture:20s}: {count:4d} images")
    print(f"{'Total':20s}: {sum(stats.values()):4d} images")


if __name__ == '__main__':
    main()

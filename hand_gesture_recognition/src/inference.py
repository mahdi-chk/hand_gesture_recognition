"""
Real-time Hand Gesture Recognition Demo
Performs inference on webcam feed with live gesture prediction.
"""

import csv
from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque, Counter
import time
from .preprocessing import HandPreprocessor
from .models import (
    GestureRecognitionCNN,
    TemporalGestureRecognitionCNNLSTM,
    create_simple_cnn_model,
    create_efficient_model,
)
import threading
import subprocess

try:
    import h5py
except Exception:
    h5py = None


def speak_text(text):
    """Announce text using system text-to-speech (Windows PowerShell)."""
    try:
        import platform
        if platform.system() == 'Windows':
            # Use Windows SAPI via PowerShell
            ps_cmd = f'Add-Type -AssemblyName System.speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'
            subprocess.Popen(['powershell', '-Command', ps_cmd], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
    except Exception:
        pass  # Silently fail if TTS not available


def load_keras_model_with_fallback(model_path):
    """Try multiple strategies to load a Keras .h5 model that may use custom classes.

    Strategies (in order):
    1. tf.keras.models.load_model with custom_objects mapping
    2. Register custom classes in keras.utils.get_custom_objects and retry
    3. Inspect HDF5 file and, if weights-only, reconstruct architecture and load weights (by_name then exact)
    """
    custom_objects = {
        'GestureRecognitionCNN': GestureRecognitionCNN,
        'TemporalGestureRecognitionCNNLSTM': TemporalGestureRecognitionCNNLSTM,
        'create_simple_cnn_model': create_simple_cnn_model,
        'create_efficient_model': create_efficient_model,
    }

    model_path = str(model_path)

    # Attempt 1: direct load with custom_objects
    try:
        print(f"[Loader] Attempt 1: Loading with custom_objects...")
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception as e1:
        print(f"[Loader] Attempt 1 failed: {type(e1).__name__}")
        first_exc = e1

    # Attempt 2: register custom objects globally and retry
    try:
        print(f"[Loader] Attempt 2: Registering custom objects globally...")
        from keras.utils import get_custom_objects

        backup = dict(get_custom_objects())
        get_custom_objects().update(custom_objects)
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"[Loader] Attempt 2 succeeded!")
            return model
        finally:
            # restore original registry
            get_custom_objects().clear()
            get_custom_objects().update(backup)
    except Exception as e2:
        print(f"[Loader] Attempt 2 failed: {type(e2).__name__}")
        pass

    # Attempt 3: inspect HDF5 file and try weight-only loading strategies
    if h5py is None:
        # h5py not available, re-raise initial exception
        print(f"[Loader] Attempt 3: h5py not available, giving up.")
        raise first_exc

    try:
        print(f"[Loader] Attempt 3: Inspecting HDF5 structure...")
        with h5py.File(model_path, 'r') as f:
            has_model_config = 'model_config' in f.attrs or 'model_config' in f
            has_layer_names = 'layer_names' in f
            print(f"[Loader]   - model_config present: {has_model_config}")
            print(f"[Loader]   - layer_names present: {has_layer_names}")

        name = Path(model_path).name.lower()

        # If file contains model config BUT no layer_names, the config is likely corrupted
        # Skip trying to load config and go straight to reconstruction
        if has_model_config and not has_layer_names:
            print(f"[Loader]   - model_config present but no layer_names (corrupted config); skipping config load")
        elif has_model_config:
            # Config looks valid, try loading directly
            try:
                print(f"[Loader] Retrying load_model with config...")
                return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            except Exception as e_cfg:
                print(f"[Loader] Config load failed: {type(e_cfg).__name__}; falling back to reconstruction")
                pass

        # Reconstruct architecture from filename heuristic
        print(f"[Loader] Attempting to reconstruct architecture from filename...")
        if 'simple' in name or 'simple_cnn' in name:
            print(f"[Loader]   - Detected: simple_cnn model")
            model = create_simple_cnn_model()
        elif 'efficient' in name or 'efficientnet' in name:
            print(f"[Loader]   - Detected: efficient model")
            model = create_efficient_model()
        elif 'cnn' in name:
            print(f"[Loader]   - Detected: cnn model")
            model = GestureRecognitionCNN()
        else:
            print(f"[Loader]   - Detected: unknown, using simple_cnn as fallback")
            model = create_simple_cnn_model()

        # Ensure model variables are created (build or dummy forward pass)
        try:
            if hasattr(model, 'build'):
                try:
                    model.build((None, 224, 224, 3))
                except Exception:
                    pass

            try:
                import numpy as _np
                _ = model(_np.zeros((1, 224, 224, 3), dtype=_np.float32), training=False)
            except Exception:
                pass

        except Exception:
            pass

        # Try loading weights by name first (more forgiving)
        try:
            print(f"[Loader] Trying load_weights(by_name=True)...")
            model.load_weights(model_path, by_name=True)
            print(f"[Loader] Attempt 3 (by_name) succeeded!")
            return model
        except Exception as e_byname:
            print(f"[Loader] by_name failed: {type(e_byname).__name__}")
            # Try exact load (may raise layer count mismatch)
            try:
                print(f"[Loader] Trying load_weights(exact)...")
                model.load_weights(model_path)
                print(f"[Loader] Attempt 3 (exact) succeeded!")
                return model
            except Exception as e_exact:
                print(f"[Loader] Exact load failed: {type(e_exact).__name__}")
                raise RuntimeError(f"Weight-loading failed (by_name: {e_byname}; exact: {e_exact})")

    except Exception:
        # If inspection/loading strategies fail, raise the first exception for context
        raise first_exc

    # If all attempts fail, re-raise the first exception for debugging
    raise first_exc


class GestureRecognitionDemo:
    """Real-time gesture recognition using webcam."""
    
    def __init__(self, model_path, class_names, image_size=(224, 224)):
        """
        Initialize demo.
        
        Args:
            model_path: Path to trained model
            class_names: List of gesture class names
            image_size: Input image size
        """
        # Load model using robust loader (handles custom classes / older h5 files)
        self.model = load_keras_model_with_fallback(model_path)
        self.class_names = class_names
        self.image_size = image_size
        self.preprocessor = HandPreprocessor(image_size=image_size)
        # Smoothing predictions using deque (shorter window for faster response)
        self.prediction_history = deque(maxlen=5)
        # Keep parallel confidence history to compute average confidence for the smoothed class
        self.confidence_history = deque(maxlen=5)
        # Announcement cooldown (seconds) to avoid repeated TTS
        self.last_announce_time = 0.0
        self.announce_cooldown = 1.2  # Reduced for faster response
        # Visual display state: persistent detected label (label, expiry_timestamp, confidence)
        self.detected_display = None
        self.display_duration = 1.2  # Show detected gesture for shorter time
        self.display_duration = 1.5  # seconds to keep big overlay visible after detection
        # Sidebar toggle
        self.show_sidebar = True
        # Prepare detections log
        self.detections_log = Path('predictions')
        self.detections_log.mkdir(parents=True, exist_ok=True)
        self.detections_csv = self.detections_log / 'detections.csv'
        if not self.detections_csv.exists():
            with open(self.detections_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'label', 'confidence'])
        
        print(f"Model loaded from {model_path}")
        print(f"Classes: {', '.join(class_names)}")
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for inference.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Preprocessed frame suitable for model input
        """
        # Try to extract hand ROI; fall back to center crop when detection fails
        roi, bbox = self.preprocessor.extract_hand_roi(frame)
        if roi is None or roi.size == 0:
            # Fallback: center crop
            h, w = frame.shape[:2]
            side = min(h, w)
            x1 = (w - side) // 2
            y1 = (h - side) // 2
            roi = frame[y1:y1+side, x1:x1+side]

        # Convert ROI BGR->RGB and resize
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(roi_rgb, self.image_size)
        # IMPORTANT: keep same numeric scale used during training (models were trained on 0-255 inputs)
        # Do NOT divide by 255 here to avoid mismatched input distributions and low-confidence outputs.
        frame_prepared = frame_resized.astype(np.float32)
        return frame_prepared
    
    def run(self, confidence_threshold=0.50, smooth_predictions=True):
        """
        Run real-time gesture recognition.
        
        Args:
            confidence_threshold: Minimum confidence to display prediction
            smooth_predictions: Use prediction smoothing
        """
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("Error: Cannot open webcam")
            return
        
        # Set camera properties
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, 30)
        
        print("\nReal-time Gesture Recognition")
        print("Controls:")
        print("  SPACE - Toggle pause/resume")
        print("  'r'   - Reset smoothing")
        print("  's'   - Save current frame")
        print("  'q'   - Quit")
        print("\n" + "="*50)
        
        paused = False
        frame_count = 0
        
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Preprocess for model
            frame_preprocessed = self.preprocess_frame(frame)
            frame_input = np.expand_dims(frame_preprocessed, axis=0)
            
            # Get prediction
            predictions = self.model.predict(frame_input, verbose=0)[0]
            pred_class = np.argmax(predictions)
            pred_confidence = predictions[pred_class]
            
            # Smooth predictions if enabled
            if smooth_predictions:
                # Append both class and confidence to histories
                self.prediction_history.append(pred_class)
                self.confidence_history.append(pred_confidence)

                # Most common class in history
                counts = Counter(self.prediction_history)
                smoothed_class, count = counts.most_common(1)[0]
                # Average confidence for that class over the history
                confs = [c for cls, c in zip(self.prediction_history, self.confidence_history) if cls == smoothed_class]
                smoothed_confidence = float(np.mean(confs)) if confs else float(pred_confidence)
            else:
                smoothed_class = pred_class
                smoothed_confidence = pred_confidence
            
            # Draw ROI rectangle
            roi_size = 300
            x1 = (w - roi_size) // 2
            y1 = (h - roi_size) // 2
            x2 = x1 + roi_size
            y2 = y1 + roi_size
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare display text
            gesture_text = self.class_names[smoothed_class]
            confidence_text = f"{smoothed_confidence:.2%}"
            
            # Determine color based on confidence
            if smoothed_confidence >= confidence_threshold:
                color = (0, 255, 0)  # Green
                status = "OK"
            else:
                color = (0, 165, 255)  # Orange
                status = "?"
            
            # Draw prediction
            cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            cv2.putText(frame, f"Confidence: {confidence_text}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, status, (w - 50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            
            # Draw top 3 predictions in a sidebar if enabled
            top_3_idx = np.argsort(predictions)[-3:][::-1]
            if self.show_sidebar:
                side_w = 300
                cv2.rectangle(frame, (w-side_w-10, 10), (w-10, 160), (30, 30, 30), -1)
                cv2.putText(frame, 'Top predictions', (w-side_w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 1)
                for i, idx in enumerate(top_3_idx):
                    text = f"{i+1}. {self.class_names[idx]}: {predictions[idx]:.1%}"
                    cv2.putText(frame, text, (w-side_w+10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            # Stable detection: if the same class appears in enough recent frames with high average confidence,
            # consider it a confirmed gesture and announce it (with cooldown)
            try:
                # For responsiveness, require 2-3 votes out of 5 and an average confidence >= 0.4
                most_common, votes = Counter(self.prediction_history).most_common(1)[0]
                avg_conf = float(np.mean([c for cls, c in zip(self.prediction_history, self.confidence_history) if cls == most_common])) if votes > 0 else 0.0
            except Exception:
                most_common, votes, avg_conf = None, 0, 0.0

            detected_label = None
            announce_threshold = max(0.45, confidence_threshold - 0.05)  # Higher threshold for better accuracy
            now = time.time()
            # Confirm when same class appears at least 2 times in history and average confidence is reasonable
            if most_common is not None and votes >= 2 and avg_conf >= announce_threshold:
                detected_label = self.class_names[most_common]
                # Only announce if enough time has passed since last announcement
                if now - self.last_announce_time > self.announce_cooldown:
                    self.last_announce_time = now
                    # Record detection in CSV log and set persistent display
                    try:
                        with open(self.detections_csv, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([datetime.utcnow().isoformat(), detected_label, f"{avg_conf:.4f}"])
                    except Exception:
                        pass
                    # Also append a human-readable log entry
                    try:
                        log_path = self.detections_log / 'detections.log'
                        with open(log_path, 'a', encoding='utf-8') as lf:
                            lf.write(f"{datetime.utcnow().isoformat()}\t{detected_label}\t{avg_conf:.4f}\n")
                    except Exception:
                        pass
                    self.detected_display = (detected_label, now + self.display_duration, avg_conf)
                    self.prediction_history.clear()
                    self.confidence_history.clear()
                    # Announce gesture in background thread
                    announce_msg = f"Detected {detected_label}"
                    thread = threading.Thread(target=speak_text, args=(announce_msg,), daemon=True)
                    thread.start()

            # If a new detection just occurred, draw a small transient overlay (primary persistent overlay below)
            if detected_label:
                cv2.rectangle(frame, (10, 10), (w-10, 60), (0, 0, 0), -1)
                cv2.putText(frame, f"{detected_label} ({avg_conf:.0%})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Draw status
            status_text = "PAUSED" if paused else "LIVE"
            status_color = (0, 0, 255) if paused else (0, 255, 0)
            cv2.putText(frame, status_text, (w - 150, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Draw help text
            cv2.putText(frame, "SPACE: Pause | s: Save | q: Quit", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # If we have a persistent detected display, render it prominently
            if self.detected_display is not None:
                label, expiry, conf = self.detected_display
                if time.time() <= expiry:
                    overlay = frame.copy()
                    y1p = int(h * 0.12)
                    y2p = int(h * 0.36)
                    cv2.rectangle(overlay, (0, y1p), (w, y2p), (0, 0, 0), -1)
                    alpha = 0.6
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    cv2.putText(frame, f"DETECTED: {label}", (int(w*0.05), int(h*0.26)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
                    cv2.putText(frame, f"Confidence: {conf:.0%}", (int(w*0.05), int(h*0.31)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
                else:
                    self.detected_display = None

            # Display frame
            cv2.imshow('Hand Gesture Recognition - Real-time', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Spacebar - pause/resume
                paused = not paused
            
            elif key == ord('s'):  # Save frame
                save_path = f'predictions/prediction_{frame_count:06d}.jpg'
                Path('predictions').mkdir(exist_ok=True)
                cv2.imwrite(save_path, frame)
                print(f"Frame saved: {save_path}")
            
            elif key == ord('r'):  # Reset smoothing
                self.prediction_history.clear()
                print("Prediction history cleared")
            
            elif key == ord('m'):
                # Toggle sidebar display
                self.show_sidebar = not self.show_sidebar
                print(f"Sidebar {'shown' if self.show_sidebar else 'hidden'}")
            
            elif key == ord('q'):  # Quit
                break
            
            if not paused:
                frame_count += 1
        
        capture.release()
        cv2.destroyAllWindows()
        print("\nDemo completed!")


class SequenceGestureRecognitionDemo:
    """Real-time gesture recognition using frame sequences (for CNN+LSTM)."""
    
    def __init__(self, model_path, class_names, sequence_length=10, image_size=(224, 224)):
        """
        Initialize sequence-based demo.
        
        Args:
            model_path: Path to trained model
            class_names: List of gesture class names
            sequence_length: Number of frames in sequence
            image_size: Input image size
        """
        # Load model using robust loader (handles custom classes / older h5 files)
        self.model = load_keras_model_with_fallback(model_path)
        self.class_names = class_names
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.preprocessor = HandPreprocessor(image_size=image_size)
        
        # Frame buffer for sequences
        self.frame_buffer = deque(maxlen=sequence_length)
        
        print(f"Model loaded from {model_path}")
        print(f"Sequence length: {sequence_length}")
        print(f"Classes: {', '.join(class_names)}")
    
    def preprocess_frame(self, frame):
        """Preprocess frame for inference - center crop for robustness."""
        # Try to extract hand ROI first
        roi, bbox = self.preprocessor.extract_hand_roi(frame, padding=30)
        
        if roi is None or roi.size == 0:
            # Fallback: use center crop (more reliable than ROI detection at inference time)
            h, w = frame.shape[:2]
            size = min(h, w)
            # Crop from center
            x1 = max(0, (w - size) // 2)
            y1 = max(0, (h - size) // 2)
            roi = frame[y1:y1+size, x1:x1+size]
        
        # Resize
        roi_resized = cv2.resize(roi, self.image_size)
        
        # Convert to RGB
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        frame_normalized = roi_rgb.astype(np.float32) / 255.0
        
        return frame_normalized
    
    def run(self, confidence_threshold=0.5):
        """Run real-time sequence-based gesture recognition."""
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("Error: Cannot open webcam")
            return
        
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, 30)
        
        print("\nSequence-based Gesture Recognition (CNN+LSTM)")
        print("Controls:")
        print("  SPACE - Toggle pause/resume")
        print("  's'   - Save current frame")
        print("  'q'   - Quit")
        
        paused = False
        
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Preprocess and add to buffer
            frame_preprocessed = self.preprocess_frame(frame)
            self.frame_buffer.append(frame_preprocessed)
            
            # Make prediction only when buffer is full
            prediction_text = ""
            confidence_text = ""
            color = (128, 128, 128)
            
            if len(self.frame_buffer) == self.sequence_length:
                # Stack frames into sequence
                sequence = np.array(list(self.frame_buffer))
                sequence_input = np.expand_dims(sequence, axis=0)
                
                # Get prediction
                predictions = self.model.predict(sequence_input, verbose=0)[0]
                pred_class = np.argmax(predictions)
                pred_confidence = predictions[pred_class]
                
                prediction_text = self.class_names[pred_class]
                confidence_text = f"{pred_confidence:.2%}"
                
                if pred_confidence >= confidence_threshold:
                    color = (0, 255, 0)
                else:
                    color = (0, 165, 255)
            
            # Draw ROI
            roi_size = 300
            x1 = (w - roi_size) // 2
            y1 = (h - roi_size) // 2
            x2 = x1 + roi_size
            y2 = y1 + roi_size
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw buffer status
            buffer_fill = len(self.frame_buffer) / self.sequence_length
            cv2.putText(frame, f"Buffer: {len(self.frame_buffer)}/{self.sequence_length}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            
            # Draw progress bar
            bar_width = 200
            bar_x = 10
            bar_y = 50
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20),
                         (100, 100, 100), -1)
            cv2.rectangle(frame, (bar_x, bar_y),
                         (int(bar_x + bar_width * buffer_fill), bar_y + 20),
                         (0, 255, 0), -1)
            
            # Draw prediction if available
            if prediction_text:
                cv2.putText(frame, f"Gesture: {prediction_text}", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                cv2.putText(frame, f"Confidence: {confidence_text}", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Display
            cv2.imshow('Sequence-based Gesture Recognition', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                paused = not paused
            elif key == ord('s'):
                save_path = 'predictions/sequence_frame.jpg'
                Path('predictions').mkdir(exist_ok=True)
                cv2.imwrite(save_path, frame)
                print(f"Frame saved: {save_path}")
            elif key == ord('q'):
                break
        
        capture.release()
        cv2.destroyAllWindows()


def main():
    """Main demo script."""
    
    print("="*60)
    print("Hand Gesture Recognition - Real-time Demo")
    print("="*60)
    
    # Try to find a saved model
    model_dir = Path('models')
    saved_models = list(model_dir.glob('*.h5'))
    
    if not saved_models:
        print("\nNo trained models found in 'models/' directory")
        print("Please train a model first using train.py")
        return
    
    print("\nAvailable models:")
    for i, model_path in enumerate(sorted(saved_models)):
        print(f"  {i+1}. {model_path.name}")
    
    # For demo, use the first available model
    model_path = sorted(saved_models)[0]
    print(f"\nUsing model: {model_path.name}")
    
    # Define class names (update based on your classes)
    class_names = ['fist', 'ok', 'palm', 'thumbs_up', 'victory']
    
    # Run demo
    # Prefer class-weighted models (most accurate, handles ok vs victory)
    preferred_names = ['weighted', 'efficient_finetuned_roi', 'efficient_finetuned', 'efficient', 'efficientnet']
    selected = None
    for name in preferred_names:
        matches = [p for p in saved_models if name in p.name.lower()]
        if matches:
            selected = sorted(matches)[-1]  # pick latest (highest numbered if multiple)
            break

    if selected is None:
        selected = sorted(saved_models)[0]

    model_path = selected
    print(f"\nUsing model: {model_path.name} (class-weighted model with 100% validation accuracy)")

    # Define class names (update based on your classes)
    class_names = ['fist', 'ok', 'palm', 'thumbs_up', 'victory']

    # Run demo with a slightly more forgiving threshold and longer smoothing
    try:
        demo = GestureRecognitionDemo(model_path, class_names)
        # increase smoothing window to reduce flicker
        demo.prediction_history = deque(maxlen=7)
        demo.run(confidence_threshold=0.4, smooth_predictions=True)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()

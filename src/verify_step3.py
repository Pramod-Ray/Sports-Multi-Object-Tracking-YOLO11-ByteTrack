"""
Step 3 Verification Script:
Verifies YOLO11n loading, ByteTrack support, and small detection-only test
on the first few frames of input/football_video.mp4 for the "person" class only.
"""
import sys
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

def main():
    print(f"Python Interpreter: {sys.executable}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Ultralytics Version: {ultralytics.__version__}")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"NumPy Version: {np.__version__}")

    # Verify ByteTrack availability
    bytetrack_available = BYTETracker is not None
    print(f"ByteTrack Available: {bytetrack_available}")

    # Load lightweight YOLO model
    model_name = "yolo11n.pt"
    print(f"Loading YOLO model: {model_name}...")
    model = YOLO(model_name)
    print("Model Loading Result: SUCCESS")

    # Verify person class index
    names = model.names
    person_class_id = None
    for k, v in names.items():
        if v == "person":
            person_class_id = k
            break
    print(f"Person class ID: {person_class_id} ('{names.get(person_class_id)}')")

    # Open video
    video_path = "input/football_video.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info: {width}x{height} @ {fps:.1f} FPS, Total Frames: {total_frames}")

    # Run SMALL detection-only test on first few frames (5 frames)
    num_test_frames = 5
    print(f"\nRunning detection-only test on first {num_test_frames} frames (detect 'person' only)...")
    
    total_persons_detected = 0
    frame_idx = 0

    for i in range(num_test_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i}")
            break
        
        frame_idx += 1
        # Inference with classes=[person_class_id]
        results = model.predict(source=frame, classes=[person_class_id], conf=0.25, verbose=False)
        result = results[0]
        boxes = result.boxes
        num_persons = len(boxes)
        total_persons_detected += num_persons
        
        confs = [f"{float(b.conf[0]):.2f}" for b in boxes] if num_persons > 0 else []
        print(f"Frame {frame_idx}: Detected {num_persons} persons (confidences: {confs[:5]}{'...' if len(confs) > 5 else ''})")

    cap.release()

    person_detection_worked = total_persons_detected > 0
    print(f"\n--- Test Summary ---")
    print(f"Frames Processed: {frame_idx}")
    print(f"Total Persons Detected: {total_persons_detected}")
    print(f"Person Detection Worked: {person_detection_worked}")
    print(f"Final Step 3 Status: {'PASS' if person_detection_worked and bytetrack_available else 'FAIL'}")

    return person_detection_worked and bytetrack_available

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

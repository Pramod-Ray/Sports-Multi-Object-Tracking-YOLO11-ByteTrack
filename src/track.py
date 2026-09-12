"""
Sports Object Tracking Pipeline - Step 4
YOLO11n Object Detection + ByteTrack Multi-Object Tracking

Usage:
    python src/track.py
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def get_color(track_id: int):
    """Generate a consistent, visually distinct BGR color for a given track ID."""
    np.random.seed(int(track_id) * 31 + 17)
    color = np.random.randint(50, 240, size=3).tolist()
    return tuple(int(c) for c in color)


def draw_label(frame, text, x1, y1, color, font_scale=0.7, thickness=2):
    """Draw a text label with a filled background rectangle for maximum readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position label above bounding box if space allows, otherwise below
    if y1 - text_h - 10 > 0:
        box_y1 = y1 - text_h - 8
        box_y2 = y1
        text_y = y1 - 4
    else:
        box_y1 = y1
        box_y2 = y1 + text_h + 8
        text_y = y1 + text_h + 4

    box_x1 = max(0, x1)
    box_x2 = min(frame.shape[1], x1 + text_w + 10)

    # Filled label background
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, -1)
    # Text in white or dark depending on color brightness
    cv2.putText(frame, text, (box_x1 + 5, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def run_tracking(
    input_path: str = "input/football_video.mp4",
    output_path: str = "output/football_tracking_bytetrack.mp4",
    model_path: str = "yolo11n.pt",
    tracker_config: str = "bytetrack.yaml",
    target_class: int = 0,
    conf_threshold: float = 0.25,
):
    print("=" * 70)
    print("STEP 4: YOLO11n + ByteTrack Sports Tracking Pipeline")
    print("=" * 70)

    # 1. Startup Validation
    print("\n[1/4] Running Startup Validation...")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  - Input Video: {input_path}")
    print(f"  - Resolution: {input_width}x{input_height}")
    print(f"  - FPS: {input_fps:.2f}")
    print(f"  - Frame Count: {total_frames}")

    # Load Model
    print(f"  - Loading YOLO model: {model_path}...")
    model = YOLO(model_path)
    print("  - YOLO model loaded successfully.")

    # Confirm ByteTrack
    print(f"  - Tracker Configuration: {tracker_config} (ByteTrack)")
    if "bytetrack" not in tracker_config.lower():
        raise ValueError(f"Tracker configuration must be ByteTrack, got: {tracker_config}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Setup VideoWriter
    fourcc_str = "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(output_path, fourcc, input_fps, (input_width, input_height))
    
    if not writer.isOpened():
        print("  - Warning: mp4v codec failed. Trying fallback 'avc1'...")
        fourcc_str = "avc1"
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(output_path, fourcc, input_fps, (input_width, input_height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to initialize VideoWriter for: {output_path}")

    print(f"  - VideoWriter opened successfully with codec: '{fourcc_str}'")
    print("  - Startup validation: PASSED")

    # 2. Tracking Execution
    print(f"\n[2/4] Processing {total_frames} frames with YOLO11n + ByteTrack...")
    
    unique_track_ids = set()
    max_simultaneous_tracks = 0
    frames_processed = 0
    start_time = time.time()
    last_log_time = start_time

    # Parameters for visual annotations scaled to high resolution (3072x1728)
    scale_factor = max(1.0, input_width / 1920.0)
    box_thickness = max(2, int(2 * scale_factor))
    font_scale = 0.6 * scale_factor
    text_thickness = max(1, int(1.8 * scale_factor))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_processed += 1

            # Run YOLO11n detection + ByteTrack tracking
            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker_config,
                classes=[target_class],
                conf=conf_threshold,
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes

            # Identify tracks in this frame
            current_visible_tracks = 0

            if boxes is not None and len(boxes) > 0:
                has_ids = boxes.id is not None
                
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    conf = float(boxes.conf[i]) if boxes.conf is not None else 0.0

                    if has_ids and boxes.id[i] is not None:
                        track_id = int(boxes.id[i])
                        unique_track_ids.add(track_id)
                        current_visible_tracks += 1
                        color = get_color(track_id)
                        label = f"Person ID {track_id} ({conf:.2f})"
                    else:
                        # Unconfirmed detection
                        color = (180, 180, 180)
                        label = f"Person ({conf:.2f})"

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
                    # Draw readable label
                    draw_label(frame, label, x1, y1, color, font_scale=font_scale, thickness=text_thickness)

            max_simultaneous_tracks = max(max_simultaneous_tracks, current_visible_tracks)

            # Draw telemetry banner
            banner_text = (
                f"YOLO11n + ByteTrack | Frame: {frames_processed}/{total_frames} | "
                f"Visible Tracks: {current_visible_tracks} | Total Unique Tracks: {len(unique_track_ids)}"
            )
            banner_bg_height = int(45 * scale_factor)
            cv2.rectangle(frame, (0, 0), (input_width, banner_bg_height), (20, 20, 20), -1)
            cv2.putText(
                frame,
                banner_text,
                (int(20 * scale_factor), int(30 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7 * scale_factor,
                (255, 255, 255),
                max(1, int(2 * scale_factor)),
                cv2.LINE_AA,
            )

            # Write annotated frame
            writer.write(frame)

            # Progress logging
            if frames_processed % 50 == 0 or frames_processed == total_frames:
                now = time.time()
                elapsed = now - start_time
                fps_current = frames_processed / elapsed if elapsed > 0 else 0
                pct = (frames_processed / total_frames) * 100.0 if total_frames > 0 else 0
                eta_s = (total_frames - frames_processed) / fps_current if fps_current > 0 else 0
                print(
                    f"  Frame {frames_processed:4d}/{total_frames} ({pct:5.1f}%) | "
                    f"Speed: {fps_current:5.1f} FPS | Elapsed: {elapsed:5.1f}s | ETA: {eta_s:5.1f}s | "
                    f"Active: {current_visible_tracks:2d} | Unique IDs: {len(unique_track_ids):2d}"
                )

    finally:
        cap.release()
        writer.release()

    total_processing_time = time.time() - start_time
    avg_processing_fps = frames_processed / total_processing_time if total_processing_time > 0 else 0.0

    print(f"\n[3/4] Processing complete!")
    print(f"  - Total Frames: {frames_processed}")
    print(f"  - Total Processing Time: {total_processing_time:.2f}s")
    print(f"  - Average Processing FPS: {avg_processing_fps:.2f} FPS")
    print(f"  - Maximum Simultaneous Tracks: {max_simultaneous_tracks}")
    print(f"  - Total Unique Track IDs: {len(unique_track_ids)}")

    # 3. Output Verification
    print("\n[4/4] Verifying Output Video...")
    output_exists = os.path.exists(output_path)
    output_size_bytes = os.path.getsize(output_path) if output_exists else 0
    output_size_mb = output_size_bytes / (1024 * 1024)

    print(f"  - Output Exists: {output_exists}")
    print(f"  - Output Size: {output_size_mb:.2f} MB ({output_size_bytes} bytes)")

    # Reopen with OpenCV
    verify_cap = cv2.VideoCapture(output_path)
    reopen_ok = verify_cap.isOpened()
    out_width = int(verify_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if reopen_ok else 0
    out_height = int(verify_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if reopen_ok else 0
    out_fps = verify_cap.get(cv2.CAP_PROP_FPS) if reopen_ok else 0.0
    out_frame_count = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if reopen_ok else 0

    print(f"  - Video Reopen Verification: {'PASS' if reopen_ok else 'FAIL'}")
    print(f"  - Output Resolution: {out_width}x{out_height}")
    print(f"  - Output FPS: {out_fps:.2f}")
    print(f"  - Output Frame Count: {out_frame_count}")

    # Read several sample frames
    sample_frames_ok = True
    for sample_idx in [0, frames_processed // 4, frames_processed // 2, frames_processed - 1]:
        verify_cap.set(cv2.CAP_PROP_POS_FRAMES, sample_idx)
        ret, frame_read = verify_cap.read()
        if not ret or frame_read is None:
            print(f"  - Warning: Failed to read sample frame at index {sample_idx}")
            sample_frames_ok = False
        else:
            print(f"  - Sample frame {sample_idx}: Successfully read (shape: {frame_read.shape})")

    verify_cap.release()

    pass_status = (
        output_exists
        and output_size_bytes > 0
        and reopen_ok
        and sample_frames_ok
        and out_frame_count == frames_processed
        and frames_processed > 0
    )

    print(f"\nFinal Step 4 Status: {'PASS' if pass_status else 'FAIL'}")

    return {
        "input_path": input_path,
        "output_path": output_path,
        "model": model_path,
        "tracker": "ByteTrack (bytetrack.yaml)",
        "target_class": "person (ID: 0)",
        "input_resolution": f"{input_width}x{input_height}",
        "input_fps": input_fps,
        "input_frame_count": total_frames,
        "output_resolution": f"{out_width}x{out_height}",
        "output_fps": out_fps,
        "output_frame_count": out_frame_count,
        "total_processing_time": total_processing_time,
        "avg_processing_fps": avg_processing_fps,
        "max_simultaneous_tracks": max_simultaneous_tracks,
        "total_unique_track_ids": len(unique_track_ids),
        "output_size_bytes": output_size_bytes,
        "output_size_mb": output_size_mb,
        "reopen_ok": reopen_ok,
        "sample_frames_ok": sample_frames_ok,
        "status": "PASS" if pass_status else "FAIL",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO11n + ByteTrack sports tracking")
    parser.add_argument("--input", default="input/football_video.mp4", help="Path to input video")
    parser.add_argument("--output", default="output/football_tracking_bytetrack.mp4", help="Path to output video")
    parser.add_argument("--model", default="yolo11n.pt", help="Path to YOLO weights")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config YAML")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    args = parser.parse_args()

    results = run_tracking(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        tracker_config=args.tracker,
        conf_threshold=args.conf,
    )

    sys.exit(0 if results["status"] == "PASS" else 1)

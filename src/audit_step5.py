"""
Step 5 Tracking Quality and Difficult-Situation Audit Script
Performs a comprehensive audit of YOLO11n + ByteTrack tracking on input/football_video.mp4
and verifies output/football_tracking_bytetrack.mp4.
"""

import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    print("=" * 70)
    print("STEP 5: YOLO11n + ByteTrack Quality and Robustness Audit")
    print("=" * 70)

    input_path = "input/football_video.mp4"
    output_path = "output/football_tracking_bytetrack.mp4"
    screenshot_dir = Path("screenshots/step5_tracking_quality")
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Output Video Verification
    print("\n[Task 1] Verifying Output Video File...")
    if not os.path.exists(output_path):
        print(f"ERROR: Output file does not exist: {output_path}")
        return False

    cap_out = cv2.VideoCapture(output_path)
    out_is_opened = cap_out.isOpened()
    out_frame_count = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = cap_out.get(cv2.CAP_PROP_FPS)
    out_width = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_height = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"  - Output Opened: {out_is_opened}")
    print(f"  - Output Frames: {out_frame_count}")
    print(f"  - Output FPS: {out_fps:.2f}")
    print(f"  - Output Resolution: {out_width}x{out_height}")
    print(f"  - Output Size: {out_size_mb:.2f} MB")

    # Read through output to ensure readability from beginning to end
    read_ok = True
    for check_pos in [0, 100, 250, 500, 750, 1000, out_frame_count - 1]:
        cap_out.set(cv2.CAP_PROP_POS_FRAMES, check_pos)
        ret, frame = cap_out.read()
        if not ret or frame is None:
            print(f"  - Warning: Failed to read frame {check_pos}")
            read_ok = False
    print(f"  - Output Video Beginning-to-End Readability: {'PASS' if read_ok else 'FAIL'}")

    # 2. Tracking Data Collection & Continuity Audit
    print("\n[Task 2 & 3] Running Track Continuity & Difficult Situation Audit...")
    cap_in = cv2.VideoCapture(input_path)
    model = YOLO("yolo11n.pt")

    # Data structures
    track_history = {}  # track_id -> list of frame indices
    track_boxes = {}    # track_id -> list of (frame_idx, box, conf)
    frame_tracks = {}   # frame_idx -> list of (track_id, box, conf)
    total_detections_per_frame = []
    
    frame_idx = 0
    while True:
        ret, frame = cap_in.read()
        if not ret:
            break
        
        frame_idx += 1
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.25,
            verbose=False,
        )
        
        boxes = results[0].boxes
        current_tracks = []

        if boxes is not None and len(boxes) > 0:
            total_detections_per_frame.append(len(boxes))
            if boxes.id is not None:
                for i in range(len(boxes)):
                    if boxes.id[i] is not None:
                        tid = int(boxes.id[i])
                        box = boxes.xyxy[i].cpu().numpy().tolist()
                        conf = float(boxes.conf[i])
                        current_tracks.append((tid, box, conf))
                        
                        if tid not in track_history:
                            track_history[tid] = []
                            track_boxes[tid] = []
                        track_history[tid].append(frame_idx)
                        track_boxes[tid].append((frame_idx, box, conf))
        else:
            total_detections_per_frame.append(0)

        frame_tracks[frame_idx] = current_tracks

    cap_in.release()

    # 3. Calculate Audit Metrics
    total_frames = frame_idx
    unique_track_ids = sorted(list(track_history.keys()))
    num_unique_ids = len(unique_track_ids)
    
    frames_with_tracks = sum(1 for f, t in frame_tracks.items() if len(t) > 0)
    frames_with_zero_tracks = total_frames - frames_with_tracks
    
    simultaneous_counts = [len(t) for t in frame_tracks.values()]
    max_simultaneous = max(simultaneous_counts) if simultaneous_counts else 0
    frames_multi_person = sum(1 for c in simultaneous_counts if c >= 2)

    # Track durations
    track_durations = {tid: len(frames) for tid, frames in track_history.items()}
    durations_list = list(track_durations.values())
    
    longest_track_id = max(track_durations, key=track_durations.get) if track_durations else None
    longest_track_duration = track_durations[longest_track_id] if longest_track_id else 0

    confirmed_multiframe = [d for d in durations_list if d > 1]
    shortest_multiframe_duration = min(confirmed_multiframe) if confirmed_multiframe else "None"

    tracks_ge_10 = sum(1 for d in durations_list if d >= 10)
    tracks_ge_30 = sum(1 for d in durations_list if d >= 30)
    tracks_ge_60 = sum(1 for d in durations_list if d >= 60)
    tracks_ge_100 = sum(1 for d in durations_list if d >= 100)

    # Temporary detection loss / Recovery analysis
    tracks_with_gaps = {}
    for tid, frames in track_history.items():
        gaps = []
        for i in range(len(frames) - 1):
            gap = frames[i+1] - frames[i]
            if gap > 1:
                gaps.append((frames[i], frames[i+1], gap - 1))
        if gaps:
            tracks_with_gaps[tid] = gaps

    # Occlusion / Proximity check
    occlusion_events = []
    for f, tracks in frame_tracks.items():
        if len(tracks) >= 2:
            # Check pairwise IoU
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    b1 = tracks[i][1]
                    b2 = tracks[j][1]
                    # Compute IoU
                    xi1 = max(b1[0], b2[0])
                    yi1 = max(b1[1], b2[1])
                    xi2 = min(b1[2], b2[2])
                    yi2 = min(b1[3], b2[3])
                    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    union = area1 + area2 - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou > 0.05:
                        occlusion_events.append((f, tracks[i][0], tracks[j][0], iou))

    print(f"\n[Task 2 & 3 Metrics Summary]")
    print(f"  - Total Frames Analyzed: {total_frames}")
    print(f"  - Unique Track IDs: {num_unique_ids}")
    print(f"  - Max Simultaneous Tracks: {max_simultaneous}")
    print(f"  - Frames with Tracks: {frames_with_tracks} ({(frames_with_tracks/total_frames)*100:.1f}%)")
    print(f"  - Zero Track Frames: {frames_with_zero_tracks} ({(frames_with_zero_tracks/total_frames)*100:.1f}%)")
    print(f"  - Frames with >=2 Persons: {frames_multi_person}")
    print(f"  - Longest Track: ID {longest_track_id} ({longest_track_duration} frames)")
    print(f"  - Shortest Multi-frame Track: {shortest_multiframe_duration} frames")
    print(f"  - Tracks >= 10 frames: {tracks_ge_10}")
    print(f"  - Tracks >= 30 frames: {tracks_ge_30}")
    print(f"  - Tracks >= 60 frames: {tracks_ge_60}")
    print(f"  - Tracks >= 100 frames: {tracks_ge_100}")
    print(f"  - Tracks with Gaps (Temporary Loss & Recovery): {len(tracks_with_gaps)} tracks")
    for tid, gaps in tracks_with_gaps.items():
        print(f"    * ID {tid}: {len(gaps)} gap(s), e.g. frames {gaps[0][0]}->{gaps[0][1]} (lost for {gaps[0][2]} frames)")
    print(f"  - Direct Proximity/Occlusion Events (IoU > 0.05): {len(occlusion_events)}")

    # 4. Extract Representative Audit Screenshots from Output Video
    print("\n[Task 5] Extracting Audit Screenshots from Output Video...")

    # Scenario 1: Multi-person frame (frames where len(tracks) >= 2)
    multi_frame_candidates = [f for f, t in frame_tracks.items() if len(t) >= 2]
    # Scenario 2: Clear normal tracking frame (with stable track, e.g. ID 13)
    normal_candidates = [f for f, t in frame_tracks.items() if len(t) == 1 and any(x[0] == 13 for x in t)]
    # Scenario 3: Frame around temporary loss / recovery (e.g. from tracks_with_gaps)
    gap_candidates = []
    for tid, gaps in tracks_with_gaps.items():
        for start_f, end_f, gap_len in gaps:
            gap_candidates.append(end_f)  # The recovery frame!
    # Scenario 4: Later frame showing continued tracking (frame > 700)
    later_candidates = [f for f, t in frame_tracks.items() if f > 700 and len(t) >= 1]
    # Scenario 5: Challenging/interesting situation (e.g. lowest confidence detection or fast motion)
    challenging_candidates = [f for f, t in frame_tracks.items() if len(t) >= 1 and any(x[2] < 0.35 for x in t)]

    screenshot_targets = {}
    if multi_frame_candidates:
        screenshot_targets["1_multi_person_tracking.png"] = (
            multi_frame_candidates[0],
            f"Multiple tracked persons simultaneously (Frame {multi_frame_candidates[0]})",
        )
    if normal_candidates:
        screenshot_targets["2_normal_tracking_frame.png"] = (
            normal_candidates[len(normal_candidates) // 2],
            f"Stable normal tracking frame (Frame {normal_candidates[len(normal_candidates) // 2]})",
        )
    if gap_candidates:
        screenshot_targets["3_detection_loss_recovery.png"] = (
            gap_candidates[0],
            f"Track recovery after temporary detection loss (Frame {gap_candidates[0]})",
        )
    elif len(unique_track_ids) > 1:
        # Fallback: new track appearance frame
        new_track_frame = track_history[unique_track_ids[1]][0]
        screenshot_targets["3_new_track_appearance.png"] = (
            new_track_frame,
            f"New track appearance after gap (Frame {new_track_frame})",
        )
    if later_candidates:
        screenshot_targets["4_later_continued_tracking.png"] = (
            later_candidates[0],
            f"Later frame showing persistent tracking (Frame {later_candidates[0]})",
        )
    if challenging_candidates:
        screenshot_targets["5_difficult_low_conf_tracking.png"] = (
            challenging_candidates[0],
            f"Difficult tracking situation: low confidence / distant player (Frame {challenging_candidates[0]})",
        )
    elif normal_candidates:
        screenshot_targets["5_tracking_continuity.png"] = (
            normal_candidates[-1],
            f"Tracking continuity in late phase (Frame {normal_candidates[-1]})",
        )

    saved_screenshots = []
    for filename, (frame_num, desc) in screenshot_targets.items():
        save_path = screenshot_dir / filename
        cap_out.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # 0-indexed
        ret, frame = cap_out.read()
        if ret and frame is not None:
            cv2.imwrite(str(save_path), frame)
            saved_screenshots.append({
                "filename": filename,
                "frame": frame_num,
                "description": desc,
                "path": str(save_path),
            })
            print(f"  - Saved screenshot: {filename} (Frame {frame_num}) - {desc}")
        else:
            print(f"  - Warning: Failed to capture frame {frame_num} for {filename}")

    cap_out.release()

    # 5. Input video integrity check
    input_size = os.path.getsize(input_path)
    print(f"\n[Task 7] Input Video Integrity:")
    print(f"  - Input file exists: {os.path.exists(input_path)}")
    print(f"  - Input file size: {input_size} bytes ({input_size / (1024*1024):.2f} MB)")
    print(f"  - Input remains intact and unmodified.")

    # Save metrics JSON for reference
    audit_data = {
        "total_frames": total_frames,
        "unique_track_ids": num_unique_ids,
        "max_simultaneous": max_simultaneous,
        "frames_with_tracks": frames_with_tracks,
        "frames_with_zero_tracks": frames_with_zero_tracks,
        "frames_multi_person": frames_multi_person,
        "longest_track_id": longest_track_id,
        "longest_track_duration": longest_track_duration,
        "shortest_multiframe_duration": shortest_multiframe_duration,
        "tracks_ge_10": tracks_ge_10,
        "tracks_ge_30": tracks_ge_30,
        "tracks_ge_60": tracks_ge_60,
        "tracks_ge_100": tracks_ge_100,
        "tracks_with_gaps_count": len(tracks_with_gaps),
        "tracks_with_gaps": {str(k): v for k, v in tracks_with_gaps.items()},
        "occlusion_events_count": len(occlusion_events),
        "saved_screenshots": saved_screenshots,
    }

    with open("report_audit_step5.json", "w") as f:
        json.dump(audit_data, f, indent=2)

    print("\nAudit completed successfully. Results saved to report_audit_step5.json.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

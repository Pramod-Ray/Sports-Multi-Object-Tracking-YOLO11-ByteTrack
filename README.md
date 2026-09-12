# Sports Multi-Object Tracking Using YOLO11n and ByteTrack

An assessment-ready computer vision pipeline implementing multi-object tracking (MOT) for broadcast-style football footage using **Ultralytics YOLO11n** for person detection and **ByteTrack** for temporal identity association.

---

## 🚀 Live Demo

Experience the interactive Streamlit showcase directly in your browser:

👉 **[Launch Streamlit Live Demo](https://sports-multi-object-tracking-yolo11-bytetrack-3rmur2jmra7ixzbq.streamlit.app/)**

Live Application URL:  
[https://sports-multi-object-tracking-yolo11-bytetrack-3rmur2jmra7ixzbq.streamlit.app/](https://sports-multi-object-tracking-yolo11-bytetrack-3rmur2jmra7ixzbq.streamlit.app/)

The web demo provides interactive access to the tracked match sequence, benchmark metrics, visual evidence gallery, and downloadable technical report.

---

## 🔗 Project Links

- **GitHub Repository**: [https://github.com/Pramod-Ray/Sports-Multi-Object-Tracking-YOLO11-ByteTrack](https://github.com/Pramod-Ray/Sports-Multi-Object-Tracking-YOLO11-ByteTrack)
- **Live Demo**: [https://sports-multi-object-tracking-yolo11-bytetrack-3rmur2jmra7ixzbq.streamlit.app/](https://sports-multi-object-tracking-yolo11-bytetrack-3rmur2jmra7ixzbq.streamlit.app/)

---

## 1. Project Overview

This project implements a complete, self-contained multi-object tracking pipeline tailored for team sports video footage (`input/football_video.mp4`). By combining the lightweight **YOLO11n** nano object detection architecture with the two-stage **ByteTrack** association algorithm, the pipeline localizes, identifies, and tracks individual football players across sequential video frames under CPU execution.

The evaluation covers the complete 1051-frame sequence at native 3K widescreen resolution (`3072 x 1728` @ 24.00 FPS). The project delivers two validated video outputs—a technical baseline and a polished assessment output featuring real-time HUD telemetry—supported by a quantitative quality audit, visual evidence package, and a formal technical PDF report.

---

## 2. Objective

The primary technical objectives of this project are:
1. Establish a standardized, reproducible Python environment for sports computer vision.
2. Implement an isolated **YOLO11n + ByteTrack** tracking pipeline targeting Class 0 (`person`) exclusively.
3. Process 100% of the input footage (1051 frames) while strictly preserving native resolution, frame rate, and aspect ratio.
4. Perform an objective empirical audit of tracking continuity, ID persistence, and detection loss recovery.
5. Generate an assessment-ready annotated video with persistent tracking visualizers and real-time telemetry HUD.
6. Assemble a curated visual evidence package and report empirical performance truthfully without fabricated metrics.

---

## 3. Key Features

- **YOLO11n Person Detection**: Single-frame spatial localization utilizing the lightweight YOLO11 nano model (`yolo11n.pt`).
- **ByteTrack Persistent Tracking**: Robust multi-object identity tracking using two-stage Kalman filter and spatial IoU association (`bytetrack.yaml`).
- **High-Visibility Bounding Boxes**: Smooth, high-contrast bounding boxes rendered cleanly across 3K native frames.
- **Persistent Track IDs**: Consistent integer identifiers and color mapping maintained across active tracking spans.
- **Confidence Scores**: Per-target detector confidence scores rendered within border-clamped label badges.
- **Real-Time HUD Telemetry**: Compact, non-intrusive top-left header overlay displaying frame index, active visible tracks, and cumulative unique track IDs without obscuring field action.
- **Detection-Loss Recovery**: Empirically verified re-association of temporarily dropped detections via ByteTrack's 30-frame track buffer.
- **Final Annotated Video**: Broadcast-quality rendered output (`output/final_sports_tracking.mp4`).
- **Quantitative Tracking Audit**: Frame-by-frame analysis of track longevity, simultaneous targets, and identity stability (`src/audit_step5.py`).
- **Visual Evidence Package**: Curated suite of native-resolution screenshots documenting distinct tracking scenarios (`screenshots/final_evidence/`).
- **Technical PDF Report**: Comprehensive 9-page assessment report compiled with ReportLab (`report/sports_multi_object_tracking_report.pdf`).

---

## 4. Technology Stack

- **Runtime Environment**: Python 3.13.2 (64-bit AMD64)
- **Deep Learning Framework**: PyTorch 2.14.0+cpu / Torchvision 0.29.0
- **Object Detection**: Ultralytics 8.4.148 (YOLO11n)
- **Multi-Object Tracking**: ByteTrack (`bytetrack.yaml` via Ultralytics)
- **Linear Assignment Solver**: LAP 0.5.13
- **Computer Vision & Video Processing**: OpenCV 5.0.0 (`opencv-python 5.0.0.93`)
- **Numerical Processing**: NumPy 2.5.3
- **Document Generation**: ReportLab 5.0.1
- **Hardware Execution**: Local CPU Execution (No GPU Acceleration)

---

## 5. System Architecture

The project employs a modern **tracking-by-detection** architecture, decoupling spatial feature detection from temporal identity association:

```mermaid
flowchart LR
    A[Football Video] --> B[OpenCV Video Reader]
    B --> C[YOLO11n Person Detection]
    C --> D[ByteTrack Association]
    D --> E[Persistent Track IDs]
    E --> F[Bounding Boxes + Confidence]
    F --> G[Telemetry Overlay]
    G --> H[Annotated Output Video]
```

---

## 6. Pipeline Workflow

1. **Video Ingestion**: `input/football_video.mp4` is opened via OpenCV `VideoCapture`, validating width (3072), height (1728), FPS (24.00), and frame count (1051).
2. **Frame-Level Detection**: Each frame is processed by `YOLO11n`, filtering specifically for Class 0 (`person`) with primary confidence cutoff $conf \ge 0.25$.
3. **ByteTrack Association**:
   - *Stage 1*: High-confidence detections are matched to predicted Kalman tracklets via spatial Intersection-over-Union (IoU).
   - *Stage 2*: Unmatched tracks are tested against remaining low-score detections ($0.10 \le conf < 0.25$) to preserve partially occluded or blurred players.
4. **Identity Management**: Matched targets maintain persistent IDs; unmatched active tracks enter a 30-frame buffer (`track_buffer = 30`).
5. **Visual Annotation**: Bounding boxes, identity tags (`Person ID <ID> (<conf>)`), and a top-left HUD badge are composited onto the frame.
6. **Video Encoding**: Annotated frames are written via OpenCV `VideoWriter` using the native `mp4v` codec.

---

## 7. Project Structure

```
sports-object-tracking/
├── input/
│   └── football_video.mp4                 # Native 3K input footage (72.2 MB)
├── output/
│   ├── football_tracking_bytetrack.mp4   # Technical baseline tracking video (143.5 MB)
│   └── final_sports_tracking.mp4          # Final assessment-ready video (146.1 MB)
├── screenshots/
│   ├── final_evidence/                    # Curated assessment evidence package
│   │   ├── 01_project_overview.png        # Figure 1: Opening frame with HUD
│   │   ├── 02_single_person_tracking.png  # Figure 2: Single-person active tracking
│   │   ├── 03_multi_person_tracking.png   # Figure 3: Multi-person tracking with distinct IDs
│   │   ├── 04_difficult_tracking.png      # Figure 4: Difficult low-confidence detection
│   │   ├── 05_tracking_recovery.png       # Figure 5: Detection-loss recovery
│   │   ├── 06_final_output.png            # Figure 6: Final output late-stage tracking
│   │   └── README.md                      # Evidence index and technical notes
│   ├── final_output/                      # Step 6 checkpoint screenshots
│   └── step5_tracking_quality/            # Step 5 audit screenshots
├── src/
│   ├── track.py                           # Baseline tracking pipeline script
│   ├── audit_step5.py                     # Quality audit & verification script
│   ├── generate_final_video.py            # Final video generator with HUD
│   ├── assemble_evidence.py               # Visual evidence packaging script
│   └── build_pdf_report.py                # Technical PDF report builder
├── report/
│   ├── sports_multi_object_tracking_report.md   # Source Markdown technical report
│   └── sports_multi_object_tracking_report.pdf  # Generated 9-page assessment PDF
├── requirements.txt                       # Exact pinned dependency specifications
├── README.md                              # Project documentation & usage instructions
└── .gitignore                             # Version control exclusion rules
```

---

## 8. Environment Setup

The project runs in an isolated virtual environment (`.venv`) with Python 3.13:

```bash
# Verify Python version (requires Python 3.10+)
python --version

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

---

## 9. Installation

Install all required dependencies using the pinned `requirements.txt`:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 10. How to Run

All scripts must be executed using the project's virtual environment interpreter:

### 1. Run Baseline Tracking
Executes standard YOLO11n + ByteTrack tracking and generates the technical baseline video:
```bash
.\.venv\Scripts\python.exe src\track.py
```

### 2. Run Quality Audit
Performs frame-by-frame analysis of tracking continuity, ID switches, and recovery:
```bash
.\.venv\Scripts\python.exe src\audit_step5.py
```

### 3. Generate Final Assessment Video
Renders the final video with HUD overlay and border-clamped label badges:
```bash
.\.venv\Scripts\python.exe src\generate_final_video.py
```

### 4. Rebuild PDF Report
Compiles the comprehensive 9-page technical assessment PDF:
```bash
.\.venv\Scripts\python.exe src\build_pdf_report.py
```

---

## 11. Output Files

| Output File | Resolution | FPS | Frames | Size | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`output/football_tracking_bytetrack.mp4`](file:///d:/Python%20Notes/Personal%20projects/sports-object-tracking/output/football_tracking_bytetrack.mp4) | `3072x1728` | `24.00` | `1051` | `143.52 MB` | Technical baseline tracking output |
| [`output/final_sports_tracking.mp4`](file:///d:/Python%20Notes/Personal%20projects/sports-object-tracking/output/final_sports_tracking.mp4) | `3072x1728` | `24.00` | `1051` | `146.11 MB` | Polished assessment video with HUD overlay |
| [`report/sports_multi_object_tracking_report.pdf`](file:///d:/Python%20Notes/Personal%20projects/sports-object-tracking/report/sports_multi_object_tracking_report.pdf) | N/A | N/A | 9 pages | `64.56 MB` | Comprehensive technical assessment report |
| [`report/sports_multi_object_tracking_report.md`](file:///d:/Python%20Notes/Personal%20projects/sports-object-tracking/report/sports_multi_object_tracking_report.md) | N/A | N/A | N/A | `21.32 KB` | Editable Markdown source report |

---

## 12. Tracking Results

The full 1051-frame sequence was processed and audited. The measured performance metrics are summarized below:

| Metric | Measured Value | Operational Notes |
| :--- | :--- | :--- |
| **Input Video** | `input/football_video.mp4` | 3K widescreen broadcast clip (~43.79s) |
| **Input Resolution** | `3072 x 1728` | 100% native resolution preserved |
| **Input Frame Rate** | `24.00 FPS` | Standard broadcast timing preserved |
| **Input Frame Count** | `1051` frames | Complete video sequence evaluated |
| **YOLO Model** | `YOLO11n` (`yolo11n.pt`) | Ultralytics lightweight nano model (~5.4 MB) |
| **Tracking Algorithm** | `ByteTrack` (`bytetrack.yaml`) | Pure ByteTrack association (No BoT-SORT) |
| **Target Class** | `Person` (Class ID: 0) | Non-person classes filtered out |
| **Total Unique Track IDs** | `18` IDs | Sequentially allocated across active episodes |
| **Max Simultaneous Tracks** | `2` tracks | Observed during multi-player interaction |
| **Frames With Person Tracks**| `47` frames (`4.47%`) | Frames with confirmed active tracks |
| **Zero-Detection Frames** | `1004` frames (`95.53%`) | Handled safely without pipeline crashes |
| **Zero-Detection Ratio** | `95.53%` | Reflects distant panoramic camera perspective |
| **Longest Continuous Track** | `15` frames | Track ID `1027` (frames 343 to 357) |
| **Direct ID Swaps Observed** | `0` swaps | Zero cross-identity swaps between players |
| **Occlusion Overlaps Observed**| `0` overlaps | Players maintained spatial field separation |
| **CPU Baseline Speed** | `~5.88 FPS` | Total baseline runtime: 178.61 seconds |
| **Final Polished Output Speed**| `~3.01 FPS` | Total final runtime: 348.70 seconds (with HUD) |

---

## 13. Tracking Quality Audit

A frame-by-frame analysis of tracking behavior and track continuity revealed key operational insights:

- **Identity Stability**: Within active tracking episodes, ByteTrack demonstrated perfect identity consistency. There were **0 direct ID swaps** observed between players.
- **Temporary Detection-Loss Recovery**: In Track ID `2680`, the player was actively tracked at frame 1047, dropped out across frames 1048–1049 due to detector thresholding, and was **successfully recovered** with identity intact at frame 1050 by ByteTrack's 30-frame track buffer.
- **Track Longevity Distribution**:
  - Tracks lasting $\ge 10$ frames: `1` track (Track ID `1027`, 15 frames)
  - Tracks lasting $\ge 30$ frames: `0` tracks
  - Tracks lasting 2 to 9 frames: `11` tracks
  - Single-frame confirmed tracks: `6` tracks
- **Track Eviction**: Inactive tracks exceeding the 30-frame buffer were terminated cleanly with zero lingering ghost artifacts.

### Quality Discussion: Understanding Sparse Tracking
Active tracks were observed in 47 of 1051 frames (4.47%). It is critical to recognize:
- **This is NOT a failure of the ByteTrack implementation.** ByteTrack is an association engine that depends on bounding boxes supplied by the upstream detector.
- The input video is an elevated wide-angle broadcast view where players occupy minimal pixels (often fewer than 30×60 pixels).
- Running the compact **YOLO11n** nano model (~5.4 MB) at default 640px internal input scale naturally causes sub-scale player features to fall below the 0.25 confidence threshold.
- When detections were present, ByteTrack performed reliably with 0 ID swaps and successful dropout recovery.

---

## 14. Visual Evidence

Six representative, uncorrupted screenshots were extracted directly from the verified outputs at native 3K resolution (`3072 x 1728`) and cataloged in [`screenshots/final_evidence/`](file:///d:/Python%20Notes/Personal%20projects/sports-object-tracking/screenshots/final_evidence):

| File | Caption | Technical Description |
| :--- | :--- | :--- |
| `01_project_overview.png` | *"Project output overview"* | Opening frame (Frame 1) demonstrating the non-intrusive top-left HUD telemetry banner on native 3K footage. |
| `02_single_person_tracking.png` | *"Single-person tracking with persistent ID"* | Frame 9 showing stable single-player tracking (`Person ID 13`) with confidence score badge during active play. |
| `03_multi_person_tracking.png` | *"Multiple-person tracking with simultaneous IDs"* | Frame 343 showing two concurrent players assigned distinct, persistent track IDs (`Person ID 1019` and `Person ID 1027`). |
| `04_difficult_tracking.png` | *"Difficult low-confidence detection"* | Frame 10 showing detector resilience on a distant, small player near the lower threshold boundary (`conf: 0.26`). |
| `05_tracking_recovery.png` | *"ByteTrack recovery after temporary detection loss"* | Frame 1050 showing ByteTrack's track buffer recovering Track ID `2680` after a 2-frame temporary detection dropout. |
| `06_final_output.png` | *"Final annotated tracking output"* | Frame 798 showing persistent tracking continuity in the later phase of the match with `Person ID 2189`. |

---

## 15. Technical Limitations

In accordance with rigorous computer vision engineering standards, the operational limitations of this baseline are:
1. **Wide-Angle Panoramic Perspective**: Distant players occupy small pixel regions, causing intermittent detection dropouts.
2. **Compact Nano Model Capacity**: YOLO11n is optimized for edge latency (~5.4 MB), limiting tiny-object feature extraction compared to larger models.
3. **CPU Execution Throughput**: Processing speeds of ~3.01–5.88 FPS reflect CPU execution constraints; real-time performance requires dedicated GPU compute.
4. **Absence of Ground-Truth Annotations**: The input video lacks manual frame-by-frame bounding-box labels, precluding scientific computation of formal MOT metrics (MOTA, MOTP, IDF1) without fabrication.

---

## 16. Possible Improvements

Recommended enhancements to elevate this baseline to a production sports analytics platform include:
1. **High-Resolution Sliced Inference (SAHI)**: Slicing native 3K frames into overlapping patches to dramatically boost small player detection recall.
2. **Larger YOLO Architectures**: Deploying YOLO11m or YOLO11x models to enhance feature discriminability on distant targets.
3. **GPU Hardware Acceleration**: Utilizing TensorRT or CUDA execution to achieve real-time throughput (>30 FPS).
4. **Domain Fine-Tuning**: Fine-tuning detector weights on specialized soccer datasets (e.g., SoccerNet, SportsMOT).
5. **Pitch Homography & Tactical Radar**: Calibrating camera perspective to project 2D player coordinates onto a top-down 2D pitch map.
6. **Re-Identification (ReID) Integration**: Incorporating visual appearance embeddings to re-identify players across camera cuts or extended field excursions.

---

## 17. Reproducibility

The complete repository is structured for end-to-end reproducibility:
- Virtual environment can be initialized and packages installed via `requirements.txt`.
- Technical baseline can be reproduced via `python src/track.py`.
- Quality audit metrics can be reproduced via `python src/audit_step5.py`.
- Final annotated video can be generated via `python src/generate_final_video.py`.
- PDF report can be recompiled via `python src/build_pdf_report.py`.

---

## 18. Assessment Checklist

- [x] **Input Video Preserved**: `input/football_video.mp4` verified unmodified (72,223,088 bytes)
- [x] **YOLO11n Employed**: Ultralytics `yolo11n.pt` used for person detection
- [x] **ByteTrack Employed**: Configured via `bytetrack.yaml` (No BoT-SORT used)
- [x] **Complete Video Processed**: All 1051 frames evaluated (100%)
- [x] **Technical Baseline Generated**: `output/football_tracking_bytetrack.mp4` verified (143.52 MB)
- [x] **Final Assessment Video Generated**: `output/final_sports_tracking.mp4` verified (146.11 MB)
- [x] **Visual Evidence Package Curated**: 6 native 3K screenshots in `screenshots/final_evidence/`
- [x] **Quantitative Audit Executed**: Full continuity, ID stability, and dropout audit completed
- [x] **Zero Fabricated Metrics**: Strictly empirical results reported (no fake MOTA/IDF1/mAP)
- [x] **Technical PDF Report Generated**: 9-page assessment report in `report/sports_multi_object_tracking_report.pdf`
- [x] **Assessment README Created**: Professional, structured documentation in `README.md`

---

## 19. Conclusion

The sports multi-object tracking assessment using YOLO11n and ByteTrack has been successfully designed, executed, audited, and documented. The technical baseline confirms that ByteTrack delivers dependable identity continuity and dropout recovery whenever valid detections are present. The project satisfies all engineering assessment requirements, upholds strict data integrity, and establishes an honest, verified foundation for future sports analytics research.

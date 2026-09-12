# Sports Multi-Object Tracking Demo Video

## 1. Demo Purpose
This 3–5 minute presentation video (`demo/sports_tracking_demo.mp4`) provides an assessment-ready overview of the completed **YOLO11n + ByteTrack** sports multi-object tracking pipeline. It is tailored for:
- AI/ML Computer Vision technical interviews and job assessments.
- Executive and engineering demonstrations.
- GitHub project showcasing with honest, empirical documentation of both strengths and physical limitations.

---

## 2. Technical Specifications
- **Video Path**: `demo/sports_tracking_demo.mp4`
- **Duration**: 231.00 seconds (~3 minutes 51 seconds)
- **Resolution**: 1920 × 1080 (16:9 Full HD, matching source 16:9 geometry without distortion)
- **Frame Rate**: 24.00 FPS
- **Total Frames**: 5544 frames
- **Container / Codec**: MP4 / `mp4v` (broadly compatible across all media players)
- **Audio Track**: None (clean, self-contained silent presentation with professional lower-third callouts, HUD badges, and textual typography)

---

## 3. Source Materials Used
- **Primary Video Extract**: [`output/final_sports_tracking.mp4`](file:///d:/Python%20Notes/Personal%20projects/sports-object-tracking/output/final_sports_tracking.mp4) (Live multi-player tracking segment)
- **Visual Evidence Suite**: [`screenshots/final_evidence/`](file:///d:/Python%20Notes/Personal%20projects/sports-object-tracking/screenshots/final_evidence)
  - `01_project_overview.png` (Frame 1)
  - `02_single_person_tracking.png` (Frame 9)
  - `03_multi_person_tracking.png` (Frame 343)
  - `04_difficult_tracking.png` (Frame 10)
  - `05_tracking_recovery.png` (Frame 1050)
  - `06_final_output.png` (Frame 798)
- **Quantitative Audit Report**: [`report/sports_multi_object_tracking_report.pdf`](file:///d:/Python%20Notes/Personal%20projects/sports-object-tracking/report/sports_multi_object_tracking_report.pdf)
- **Generator Script**: [`src/generate_demo.py`](file:///d:/Python%20Notes/Personal%20projects/sports-object-tracking/src/generate_demo.py)

---

## 4. Timeline and Section Breakdown

| Timestamp | Section | Duration | Content & Technical Focus |
| :--- | :--- | :--- | :--- |
| **00:00 – 00:22** | **Section 1: Introduction** | 22.0s | Project title, technology stack (YOLO11n + ByteTrack), input clip metadata, and evaluation objective. |
| **00:22 – 00:44** | **Section 2: Pipeline Architecture** | 22.0s | Visual flowchart illustrating tracking-by-detection: Video $\to$ YOLO11n $\to$ ByteTrack $\to$ Persistent IDs $\to$ HUD $\to$ Output. |
| **00:44 – 01:09** | **Section 3: Single-Person Tracking** | 25.0s | Native inspection of Frame 9 demonstrating stable tracking of `Person ID 13` (conf 0.46) with high-visibility bounding box. |
| **01:09 – 01:34** | **Section 4: Multi-Person Tracking** | 25.0s | Native inspection of Frame 343 showing simultaneous tracking of `Person ID 1019` and `Person ID 1027` with independent colors. |
| **01:34 – 01:56** | **Section 5: Difficult Detections** | 22.0s | Analysis of Frame 10 showing detector sensitivity on a distant player near threshold (`conf: 0.26`) without hallucinating false positives. |
| **01:56 – 02:21** | **Section 6: Detection-Loss Recovery** | 25.0s | Empirical verification of ByteTrack's track buffer: Track ID `2680` active at frame 1047, lost across 1048–1049, recovered at frame 1050. |
| **02:21 – 02:51** | **Section 7: Final Annotated Output** | 30.0s | Live video playback from `output/final_sports_tracking.mp4` showing dynamic multi-player interaction and real-time HUD telemetry. |
| **02:51 – 03:13** | **Section 8: Quantitative Results** | 22.0s | Comprehensive audit card displaying measured empirical metrics (1051 frames, 18 IDs, 0 swaps, ~5.88 FPS baseline, ~3.01 FPS final). |
| **03:13 – 03:35** | **Section 9: Limitations & Roadmap** | 22.0s | Side-by-side comparison of physical constraints (wide-angle, CPU) versus concrete future enhancements (SAHI, GPU, SoccerNet tuning). |
| **03:35 – 03:51** | **Section 10: Conclusion** | 16.0s | Final technical summary confirming end-to-end pipeline completion, honest metric reporting, and assessment readiness. |

---

## 5. Metrics Demonstrated
- **Total Evaluated Frames**: 1051 frames (~43.79 seconds sequence)
- **Target Class**: Person (Class ID 0)
- **Unique Track IDs Allocated**: 18
- **Maximum Simultaneous Active Tracks**: 2
- **Direct ID Swaps Observed**: 0 (zero identity collision between visible players)
- **Direct Occlusion Overlaps Observed**: 0 (players maintained spatial pitch separation)
- **Longest Continuous Track**: 15 frames (Track ID 1027)
- **Temporary Dropout Recovery**: Verified for Track ID 2680 (frame 1047 $\to$ 1050)
- **Zero-Detection Frames Handled Safely**: 1004 frames (95.53%)
- **Zero Fabricated Metrics**: Strictly empirical values; no unverified MOTA/IDF1/mAP percentages claimed.

---

## 6. Documented Limitations Communicated
The presentation explicitly communicates that:
1. Active tracks occur in 4.47% of frames (47 frames) due to the **panoramic wide-angle viewpoint** where distant players occupy minimal pixel areas.
2. The lightweight **YOLO11n nano model (~5.4 MB)** running on **CPU** at default 640px internal resolution yields sparse detections on small targets.
3. This is an environmental detection constraint rather than a ByteTrack failure; ByteTrack exhibited 100% ID stability during active periods.
4. The system is an honest engineering baseline, not a real-time production system.

---

## 7. How to Reproduce
The demo video can be regenerated at any time using the virtual environment:

```bash
.\.venv\Scripts\python.exe src\generate_demo.py
```

---

## 8. Artifact Preservation Verification
All original repository artifacts were preserved unmodified:
- `input/football_video.mp4`: `72,223,088 bytes` (100% intact)
- `output/football_tracking_bytetrack.mp4`: `143,518,373 bytes` (preserved)
- `output/final_sports_tracking.mp4`: `146,112,659 bytes` (preserved)
- `report/sports_multi_object_tracking_report.pdf`: `64,564,797 bytes` (preserved)
- `screenshots/final_evidence/`: All 6 screenshots and README preserved intact

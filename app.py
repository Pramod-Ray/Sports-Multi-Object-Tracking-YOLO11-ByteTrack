"""
Sports Multi-Object Tracking Showcase Application
Built with Streamlit.

Lightweight demonstration showcase for YOLO11n + ByteTrack sports tracking pipeline.
No YOLO inference or model weights are loaded at runtime.
"""

from pathlib import Path
import streamlit as st

# Anchor all paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_VIDEO_PATH = PROJECT_ROOT / "demo" / "sports_tracking_demo_web.mp4"
EVIDENCE_DIR = PROJECT_ROOT / "screenshots" / "final_evidence"
REPORT_PDF_PATH = PROJECT_ROOT / "report" / "sports_multi_object_tracking_report.pdf"
GITHUB_REPO_URL = "https://github.com/Pramod-Ray/Sports-Multi-Object-Tracking-YOLO11-ByteTrack"

# Page Configuration
st.set_page_config(
    page_title="Sports Multi-Object Tracking | YOLO11n + ByteTrack",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive presentation styling
st.markdown(
    """
    <style>
    /* Metric card styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(240, 244, 248, 0.8) 0%, rgba(255, 255, 255, 0.9) 100%);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.08);
        transition: all 0.2s ease-in-out;
    }
    
    /* Section card container */
    .showcase-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    
    /* Subtle tags & badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.025em;
        text-transform: uppercase;
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        margin-bottom: 8px;
    }
    
    /* Header subtitle styling */
    .hero-subtitle {
        font-size: 1.15rem;
        color: #475569;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation & Information
with st.sidebar:
    st.markdown("### ⚽ Project Navigation")
    st.markdown(
        """
        - [Project Overview](#project-overview)
        - [Benchmark & Audit Metrics](#benchmark-metrics)
        - [Tracking Video Demonstration](#tracking-demonstration)
        - [Tracking Evidence Gallery](#tracking-evidence)
        - [Technical Report](#technical-report)
        - [GitHub Repository](#github-repository)
        """
    )
    st.divider()

    st.markdown("### 🛠️ Architecture Stack")
    st.markdown(
        """
        - **Detector**: Ultralytics YOLO11n (`yolo11n.pt`)
        - **Tracker**: ByteTrack (`bytetrack.yaml`)
        - **Target Class**: `person` (ID 0)
        - **Association**: Kalman Filter + Bipartite Hungarian Matching
        - **Inference Mode**: Offline showcase / No runtime weights
        """
    )
    st.divider()

    st.markdown("### 🎓 Submission Context")
    st.caption(
        "Sports Multi-Object Tracking Assessment Showcase. Built with clean, "
        "production-grade components for reproducible visual verification."
    )

# Header Section
st.markdown('<span class="badge">Computer Vision & Deep Learning</span>', unsafe_allow_html=True)
st.title("Sports Multi-Object Tracking Using YOLO11n and ByteTrack")
st.markdown(
    '<p class="hero-subtitle">'
    "YOLO11n is used for person detection and ByteTrack for multi-object tracking."
    "</p>",
    unsafe_allow_html=True,
)

# Section: Project Overview
st.markdown('<a name="project-overview"></a>', unsafe_allow_html=True)
st.header("📌 Project Overview")
overview_col1, overview_col2 = st.columns([3, 2])

with overview_col1:
    st.markdown(
        """
        This project implements a high-performance **Multi-Object Tracking (MOT)** pipeline 
        engineered specifically for high-resolution sports footage. 
        
        Using native **3072 × 1728 (3K)** broadcast football footage, the architecture pairs 
        **Ultralytics YOLO11n** for real-time person localization with the **ByteTrack** association 
        algorithm to maintain identity persistence across fast-paced camera pans, player occlusions, 
        and scale variations.

        **Key Architectural Highlights:**
        - **Two-Stage Detection Association**: ByteTrack retains both high-confidence and low-confidence 
          detection boxes, matching true detections in difficult conditions where conventional trackers fail.
        - **Kalman Filter Motion Prediction**: Predicts future bounding box trajectory to bridge temporary 
          detection dropouts (buffered recovery).
        - **Non-Intrusive HUD Overlay**: Telemetry dashboard showing model details, live frame counter, 
          instantaneous active tracks, and cumulative unique track counts.
        """
    )

with overview_col2:
    st.info(
        """
        **System Architecture Flow:**
        1. **Raw Video Input**: `input/football_video.mp4` (24 FPS, 1051 frames)
        2. **Detector Stage**: YOLO11n (`person` class, conf ≥ 0.25, IOU 0.70)
        3. **Association Stage**: ByteTrack matching cascade (Hungarian algorithm)
        4. **Track Buffer**: 30-frame persistence buffer for lost tracks
        5. **Visual Output**: Professional 3K video render with HUD telemetry
        """
    )

st.divider()

# Section: Metrics
st.markdown('<a name="benchmark-metrics"></a>', unsafe_allow_html=True)
st.header("📊 Benchmark & Audit Metrics")
st.markdown(
    "Audited performance parameters and tracking metrics recorded across the full match sequence:"
)

# Metrics Grid - Row 1: Video Specifications
st.subheader("Video Specifications")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Input Video", "football_video.mp4")
with col2:
    st.metric("Frames", "1051")
with col3:
    st.metric("Resolution", "3072 × 1728")
with col4:
    st.metric("FPS", "24")

# Metrics Grid - Row 2: Tracking Precision & Stability
st.subheader("Tracking Precision & Consistency")
col5, col6, col7 = st.columns(3)
with col5:
    st.metric("Unique Tracked IDs", "18", help="Total cumulative unique person IDs assigned across 1051 frames")
with col6:
    st.metric("Maximum Simultaneous IDs", "2", help="Peak concurrent players tracked within a single frame")
with col7:
    st.metric("Direct ID Swaps", "0", help="Zero identity swaps detected between adjacent active players")

# Metrics Grid - Row 3: Processing Speed
st.subheader("Processing Speed Benchmarks")
col8, col9 = st.columns(2)
with col8:
    st.metric(
        "Baseline Processing Speed",
        "approximately 5.88 FPS",
        help="Raw detection and ByteTrack tracking loop without telemetry rendering",
    )
with col9:
    st.metric(
        "Final Processing Speed",
        "approximately 3.01 FPS",
        help="Full production pipeline including 3K HUD telemetry drawing, bounding boxes, and video encoding",
    )

st.divider()

# Section: Final Demo Video
st.markdown('<a name="tracking-demonstration"></a>', unsafe_allow_html=True)
st.header("🎥 Tracking Demonstration")
st.markdown(
    "Final compiled tracking demonstration featuring the full HUD telemetry overlay, bounding boxes, "
    "track IDs, and confidence badges on native footage:"
)

if DEMO_VIDEO_PATH.exists():
    try:
        st.video(str(DEMO_VIDEO_PATH))
        st.caption(
            f"Displaying: `{DEMO_VIDEO_PATH.relative_to(PROJECT_ROOT)}` "
            "(Processed with YOLO11n + ByteTrack at 24 FPS)"
        )
    except Exception as e:
        st.error(f"Error loading video file: {e}")
else:
    st.warning(
        f"Demo video not found at expected path: `{DEMO_VIDEO_PATH.relative_to(PROJECT_ROOT)}`\n\n"
        "Please verify that the demo video has been generated or placed in the `demo/` directory."
    )

st.divider()

# Section: Tracking Evidence
st.markdown('<a name="tracking-evidence"></a>', unsafe_allow_html=True)
st.header("🔍 Tracking Evidence")
st.markdown(
    "Key frames highlighting tracker performance, multi-person identification, low-confidence resilience, "
    "and detection loss recovery:"
)

# Evidence metadata lookup for contextual captions
EVIDENCE_METADATA = {
    "01_project_overview.png": {
        "title": "1. Project Overview & HUD Overlay",
        "caption": "Opening frame (Frame 1) demonstrating clean HUD telemetry overlay (model, frame counter, active tracks, unique tracks) on native 3K footage.",
    },
    "02_single_person_tracking.png": {
        "title": "2. Single-Person Tracking Stability",
        "caption": "Persistent single-player tracking (Person ID 13) with high-visibility bounding box and confidence score badge during active play.",
    },
    "03_multi_person_tracking.png": {
        "title": "3. Multi-Person Tracking & Distinction",
        "caption": "Multi-object tracking (Frame 343) showing two concurrent players assigned distinct, persistent track IDs (Person ID 1019 & 1027) with distinct color coding.",
    },
    "04_difficult_tracking.png": {
        "title": "4. Distant & Low-Confidence Detection",
        "caption": "Detector and tracker resilience on a distant player near the lower confidence threshold boundary (conf 0.26) without false positives.",
    },
    "05_tracking_recovery.png": {
        "title": "5. Detection Loss & Track Recovery",
        "caption": "ByteTrack track buffer successfully recovering Track ID 2680 after a temporary 2-frame detection dropout (frames 1048–1049) at Frame 1050.",
    },
    "06_final_output.png": {
        "title": "6. Late-Stage Match Tracking Continuity",
        "caption": "Stable tracking continuity in later phase of match (Frame 798) displaying active Track ID 2189 and updated cumulative unique ID counts.",
    },
}

if EVIDENCE_DIR.exists() and EVIDENCE_DIR.is_dir():
    # Discover all image files (.png, .jpg, .jpeg)
    image_files = sorted(
        [
            f
            for f in EVIDENCE_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]
    )

    if image_files:
        # Display images in an organized 2-column layout
        for i in range(0, len(image_files), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(image_files):
                    img_path = image_files[i + j]
                    meta = EVIDENCE_METADATA.get(
                        img_path.name,
                        {
                            "title": img_path.stem.replace("_", " ").title(),
                            "caption": f"Evidence frame from {img_path.name}",
                        },
                    )
                    with cols[j]:
                        st.markdown(f"##### {meta['title']}")
                        st.image(str(img_path), use_container_width=True)
                        st.caption(meta["caption"])
                        st.write("")
    else:
        st.info("No screenshot images found in `screenshots/final_evidence/`.")
else:
    st.warning(
        f"Evidence directory not found at: `{EVIDENCE_DIR.relative_to(PROJECT_ROOT)}`. "
        "Visual evidence screenshots will appear here once available."
    )

st.divider()

# Section: Technical Report
st.markdown('<a name="technical-report"></a>', unsafe_allow_html=True)
st.header("📄 Technical Report")
st.markdown(
    """
    A comprehensive engineering report is available documenting the complete tracking methodology, 
    quantitative step-by-step audit, mathematical formulation of ByteTrack association, 
    and performance trade-off analysis.
    """
)

if REPORT_PDF_PATH.exists():
    file_size_mb = REPORT_PDF_PATH.stat().st_size / (1024 * 1024)
    with open(REPORT_PDF_PATH, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    rep_col1, rep_col2 = st.columns([2, 1])
    with rep_col1:
        st.markdown(
            f"""
            - **Document**: `sports_multi_object_tracking_report.pdf`
            - **Size**: approximately {file_size_mb:.1f} MB
            - **Contents**: Full experimental evaluation, baseline vs final speed comparison, 
              occlusion and ID switch analysis, architectural decisions, and visual exhibits.
            """
        )
    with rep_col2:
        st.download_button(
            label="📥 Download Technical Report (PDF)",
            data=pdf_bytes,
            file_name="sports_multi_object_tracking_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
else:
    st.warning(
        f"Technical report not found at expected location: `{REPORT_PDF_PATH.relative_to(PROJECT_ROOT)}`\n\n"
        "Ensure `report/sports_multi_object_tracking_report.pdf` has been generated."
    )

st.divider()

# Section: GitHub Repository
st.markdown('<a name="github-repository"></a>', unsafe_allow_html=True)
st.header("🔗 GitHub Repository")
st.markdown(
    """
    Access the complete codebase, configuration files, reproduction steps, and documentation 
    on the official project GitHub repository:
    """
)

gh_col1, gh_col2 = st.columns([3, 1])
with gh_col1:
    st.markdown(
        f"**Repository**: [{GITHUB_REPO_URL}]({GITHUB_REPO_URL})\n\n"
        "Features automated tracking pipelines, modularized video annotation utilities, "
        "and reproducible test suites."
    )
with gh_col2:
    st.link_button(
        label="⭐ View on GitHub",
        url=GITHUB_REPO_URL,
        use_container_width=True,
    )

st.markdown("---")
st.caption(
    "Sports Multi-Object Tracking Showcase • Developed with YOLO11n & ByteTrack • "
    "Designed for Internship Assessment Submission"
)

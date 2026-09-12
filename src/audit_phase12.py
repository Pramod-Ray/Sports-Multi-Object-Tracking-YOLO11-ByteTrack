"""
Phase 12: Final Quality Check & Submission Readiness Audit Script
Non-destructive validation of all completed project artifacts and the submission ZIP.
"""

import os
import sys
import cv2
import zipfile
import json
import re
from pathlib import Path

def main():
    print("=" * 70)
    print("PHASE 12 COMPREHENSIVE SUBMISSION READINESS AUDIT")
    print("=" * 70)

    # 1. Project Structure
    files_to_check = [
        "input/football_video.mp4",
        "output/football_tracking_bytetrack.mp4",
        "output/final_sports_tracking.mp4",
        "screenshots/step5_tracking_quality/1_multi_person_tracking.png",
        "screenshots/final_output/final_output_tracking.png",
        "screenshots/final_evidence/01_project_overview.png",
        "screenshots/final_evidence/02_single_person_tracking.png",
        "screenshots/final_evidence/03_multi_person_tracking.png",
        "screenshots/final_evidence/04_difficult_tracking.png",
        "screenshots/final_evidence/05_tracking_recovery.png",
        "screenshots/final_evidence/06_final_output.png",
        "screenshots/final_evidence/README.md",
        "report/sports_multi_object_tracking_report.pdf",
        "report/sports_multi_object_tracking_report.md",
        "demo/sports_tracking_demo.mp4",
        "demo/README.md",
        "README.md",
        "requirements.txt",
        ".gitignore",
        "src/track.py",
        "src/audit_step5.py",
        "src/generate_final_video.py",
        "src/assemble_evidence.py",
        "src/build_pdf_report.py",
        "src/generate_demo.py",
        "src/package_submission.py",
    ]
    c1_pass = all(os.path.exists(f) for f in files_to_check)
    print(f"CHECK 1 (Project Structure): {'PASS' if c1_pass else 'FAIL'}")

    # 2. Tracking Artifact Consistency
    # Check that documented values match audit json
    audit_json_path = "report_audit_step5.json"
    c2_pass = False
    if os.path.exists(audit_json_path):
        with open(audit_json_path) as f:
            adata = json.load(f)
            c2_pass = (
                adata.get("total_frames") == 1051
                and adata.get("unique_track_ids") == 18
                and adata.get("max_simultaneous") == 2
                and adata.get("frames_with_tracks") == 47
                and adata.get("frames_with_zero_tracks") == 1004
                and adata.get("longest_track_duration") == 15
                and "2680" in adata.get("tracks_with_gaps", {})
            )
    print(f"CHECK 2 (Artifact Consistency): {'PASS' if c2_pass else 'FAIL'}")

    # 3. Output Video Validation
    videos = {
        "Baseline": ("output/football_tracking_bytetrack.mp4", 3072, 1728, 24.0, 1051),
        "Final": ("output/final_sports_tracking.mp4", 3072, 1728, 24.0, 1051),
        "Demo": ("demo/sports_tracking_demo.mp4", 1920, 1080, 24.0, 5544),
    }
    c3_pass = True
    for vname, (vpath, exp_w, exp_h, exp_fps, exp_frames) in videos.items():
        if not os.path.exists(vpath):
            print(f"  [-] Missing: {vpath}")
            c3_pass = False
            continue
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  [-] Cannot open: {vpath}")
            c3_pass = False
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = round(cap.get(cv2.CAP_PROP_FPS), 2)
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        cap.release()
        valid = (w == exp_w and h == exp_h and fps == exp_fps and cnt == exp_frames and ret and frame is not None)
        if not valid:
            c3_pass = False
        print(f"  - {vname} Video: {w}x{h}, {fps} FPS, {cnt} frames | Readable: {ret} [{'PASS' if valid else 'FAIL'}]")
    print(f"CHECK 3 (Video Validation): {'PASS' if c3_pass else 'FAIL'}")

    # 4. Screenshot Validation
    evidence_files = [
        "01_project_overview.png",
        "02_single_person_tracking.png",
        "03_multi_person_tracking.png",
        "04_difficult_tracking.png",
        "05_tracking_recovery.png",
        "06_final_output.png",
    ]
    c4_pass = True
    for ef in evidence_files:
        epath = os.path.join("screenshots/final_evidence", ef)
        if not os.path.exists(epath):
            c4_pass = False
            continue
        img = cv2.imread(epath)
        if img is None or img.shape != (1728, 3072, 3):
            c4_pass = False
    print(f"CHECK 4 (Screenshot Validation): {'PASS' if c4_pass else 'FAIL'} (6/6 native 3K screenshots verified)")

    # 5. Report Validation
    rep_pdf = "report/sports_multi_object_tracking_report.pdf"
    rep_md = "report/sports_multi_object_tracking_report.md"
    page_count = 0
    with open(rep_pdf, "rb") as f:
        pdf_bytes = f.read()
        pages = re.findall(rb"/Type\s*/Page[^s]", pdf_bytes)
        page_count = len(pages)
    c5_pass = os.path.exists(rep_pdf) and os.path.getsize(rep_pdf) > 0 and page_count == 9 and os.path.exists(rep_md)
    print(f"CHECK 5 (Report Validation): {'PASS' if c5_pass else 'FAIL'} (PDF: {page_count} pages, {len(pdf_bytes)} bytes)")

    # 6. README Validation
    readme_text = open("README.md", encoding="utf-8").read()
    sections = [
        "1. Project Overview",
        "2. Objective",
        "3. Key Features",
        "4. Technology Stack",
        "5. System Architecture",
        "6. Pipeline Workflow",
        "7. Project Structure",
        "8. Environment Setup",
        "9. Installation",
        "10. How to Run",
        "11. Output Files",
        "12. Tracking Results",
        "13. Tracking Quality Audit",
        "14. Visual Evidence",
        "15. Technical Limitations",
        "16. Possible Improvements",
        "17. Reproducibility",
        "18. Assessment Checklist",
        "19. Conclusion",
    ]
    c6_pass = all(s in readme_text for s in sections)
    print(f"CHECK 6 (README Validation): {'PASS' if c6_pass else 'FAIL'} (19/19 sections present)")

    # 7. Demo Validation
    demo_path = "demo/sports_tracking_demo.mp4"
    demo_readme = "demo/README.md"
    c7_pass = os.path.exists(demo_path) and os.path.exists(demo_readme) and os.path.getsize(demo_path) > 0
    print(f"CHECK 7 (Demo Validation): {'PASS' if c7_pass else 'FAIL'}")

    # 8. Requirements Validation
    req_text = open("requirements.txt", encoding="utf-8").read()
    core_pkgs = ["ultralytics==8.4.148", "opencv-python==5.0.0.93", "numpy==2.5.3", "torch==2.14.0", "lap==0.5.13"]
    c8_pass = all(pkg in req_text for pkg in core_pkgs)
    print(f"CHECK 8 (Requirements Validation): {'PASS' if c8_pass else 'FAIL'}")

    # 9. Safety & Cleanliness
    clean = True
    for root, dirs, files in os.walk("."):
        if ".venv" in root or ".git" in root:
            continue
        for f in files:
            if f.endswith(".pyc") or f.endswith(".log") or f == "Thumbs.db":
                print(f"  [-] Unwanted file found: {os.path.join(root, f)}")
                clean = False
    c9_pass = clean
    print(f"CHECK 9 (Safety/Cleanliness): {'PASS' if c9_pass else 'FAIL'}")

    # 10. ZIP Validation
    zip_path = "D:/Python Notes/Personal projects/sports-object-tracking-submission.zip"
    c10_pass = False
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            znames = zf.namelist()
            has_reqs = any("requirements.txt" in n for n in znames)
            has_in = any("football_video.mp4" in n for n in znames)
            has_out = any("final_sports_tracking.mp4" in n for n in znames)
            has_demo = any("sports_tracking_demo.mp4" in n for n in znames)
            has_pdf = any("sports_multi_object_tracking_report.pdf" in n for n in znames)
            no_venv = not any(".venv" in n for n in znames)
            no_pycache = not any("__pycache__" in n for n in znames)
            c10_pass = has_reqs and has_in and has_out and has_demo and has_pdf and no_venv and no_pycache
            print(f"CHECK 10 (ZIP Validation): {'PASS' if c10_pass else 'FAIL'} (Size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB, {len(znames)} items)")
    else:
        print("CHECK 10 (ZIP Validation): FAIL (Missing ZIP)")

    # 11. Overall Status
    all_checks = [c1_pass, c2_pass, c3_pass, c4_pass, c5_pass, c6_pass, c7_pass, c8_pass, c9_pass, c10_pass]
    overall_status = "PASS — READY FOR SUBMISSION" if all(all_checks) else "FAIL — FIX REQUIRED"
    print(f"\nCHECK 11 (Assessment Readiness): {overall_status}")
    return overall_status == "PASS — READY FOR SUBMISSION"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

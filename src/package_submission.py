"""
Phase 11: Final GitHub and ZIP Packaging Script
Creates D:/Python Notes/Personal projects/sports-object-tracking-submission
and D:/Python Notes/Personal projects/sports-object-tracking-submission.zip
without modifying the original project artifacts.
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path


def main():
    print("=" * 70)
    print("PHASE 11: Final Packaging for GitHub & ZIP Submission")
    print("=" * 70)

    src_root = Path("D:/Python Notes/Personal projects/sports-object-tracking")
    submission_root = Path("D:/Python Notes/Personal projects/sports-object-tracking-submission")
    dest_project = submission_root / "sports-object-tracking"
    zip_path = Path("D:/Python Notes/Personal projects/sports-object-tracking-submission.zip")

    # Clean prior packaging if any
    if submission_root.exists():
        shutil.rmtree(submission_root)
    if zip_path.exists():
        zip_path.unlink()

    dest_project.mkdir(parents=True, exist_ok=True)
    print(f"Created submission directory: {dest_project}")

    # Explicit whitelist of directories and files to include
    dirs_to_copy = [
        "input",
        "output",
        "screenshots/step5_tracking_quality",
        "screenshots/final_output",
        "screenshots/final_evidence",
        "src",
        "report",
        "demo",
    ]

    files_to_copy = [
        "requirements.txt",
        "README.md",
        ".gitignore",
        "yolo11n.pt",
        "report_audit_step5.json",
    ]

    # Copy files
    for f in files_to_copy:
        s_file = src_root / f
        d_file = dest_project / f
        if s_file.exists():
            shutil.copy2(s_file, d_file)
            print(f"  [+] Copied file: {f}")

    # Copy directories (ignoring __pycache__, .venv, and .pyc)
    def ignore_patterns(path, names):
        ignored = set()
        for name in names:
            if name in [".venv", "__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea", ".vscode"]:
                ignored.add(name)
            elif name.endswith(".pyc") or name.endswith(".pyo") or name.endswith(".log"):
                ignored.add(name)
        return ignored

    for d in dirs_to_copy:
        s_dir = src_root / d
        d_dir = dest_project / d
        if s_dir.exists():
            shutil.copytree(s_dir, d_dir, ignore=ignore_patterns)
            print(f"  [+] Copied tree: {d}")

    # Verify copied files
    copied_files = [p for p in dest_project.rglob("*") if p.is_file()]
    total_package_size = sum(p.stat().st_size for p in copied_files)

    input_vid = dest_project / "input/football_video.mp4"
    final_vid = dest_project / "output/final_sports_tracking.mp4"
    demo_vid = dest_project / "demo/sports_tracking_demo.mp4"
    pdf_rep = dest_project / "report/sports_multi_object_tracking_report.pdf"

    print(f"\nPackage Inventory:")
    print(f"  - Total Copied Files: {len(copied_files)}")
    print(f"  - Total Package Size: {total_package_size / (1024*1024):.2f} MB ({total_package_size} bytes)")
    print(f"  - Input Video Size: {input_vid.stat().st_size / (1024*1024):.2f} MB ({input_vid.stat().st_size} bytes)")
    print(f"  - Final Tracking Video Size: {final_vid.stat().st_size / (1024*1024):.2f} MB ({final_vid.stat().st_size} bytes)")
    print(f"  - Demo Video Size: {demo_vid.stat().st_size / (1024*1024):.2f} MB ({demo_vid.stat().st_size} bytes)")
    print(f"  - PDF Report Size: {pdf_rep.stat().st_size / (1024*1024):.2f} MB ({pdf_rep.stat().st_size} bytes)")

    # Build ZIP Archive
    print(f"\nBuilding ZIP Archive at: {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in copied_files:
            rel_path = p.relative_to(submission_root)
            zf.write(p, arcname=str(rel_path))

    zip_size = zip_path.stat().st_size
    print(f"ZIP Archive Created Successfully! Size: {zip_size / (1024*1024):.2f} MB ({zip_size} bytes)")

    # Verify ZIP Archive
    print("\nVerifying ZIP Archive Contents...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()

        checks = {
            "README.md": any("README.md" in n and "sports-object-tracking/README.md" == n.replace("\\", "/") for n in namelist),
            "requirements.txt": any("sports-object-tracking/requirements.txt" == n.replace("\\", "/") for n in namelist),
            "src/": any("sports-object-tracking/src/" in n.replace("\\", "/") for n in namelist),
            "input/football_video.mp4": any("sports-object-tracking/input/football_video.mp4" == n.replace("\\", "/") for n in namelist),
            "output/final_sports_tracking.mp4": any("sports-object-tracking/output/final_sports_tracking.mp4" == n.replace("\\", "/") for n in namelist),
            "report/sports_multi_object_tracking_report.pdf": any("sports-object-tracking/report/sports_multi_object_tracking_report.pdf" == n.replace("\\", "/") for n in namelist),
            "demo/sports_tracking_demo.mp4": any("sports-object-tracking/demo/sports_tracking_demo.mp4" == n.replace("\\", "/") for n in namelist),
            "screenshots/final_evidence/": any("sports-object-tracking/screenshots/final_evidence/" in n.replace("\\", "/") for n in namelist),
            ".venv excluded": not any(".venv" in n for n in namelist),
            "__pycache__ excluded": not any("__pycache__" in n for n in namelist),
        }

        for check_name, passed in checks.items():
            print(f"  - Check '{check_name}': {'PASS' if passed else 'FAIL'}")

    all_passed = all(checks.values())
    print(f"\nFinal Phase 11 Status: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python
"""
Install a face recognition backend so the monitoring system can run.

- Tries InsightFace first (best accuracy).
- If that fails (e.g. missing Microsoft C++ Build Tools on Windows),
  installs DeepFace instead (no compilation, works on Windows).

Run: python install_face_recognition.py
"""

import subprocess
import sys


def run(cmd: list) -> bool:
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install"] + cmd,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stderr or r.stdout)
    return r.returncode == 0


def main():
    print("Installing face recognition backend...")
    print()

    if run(["insightface>=0.7.3"]):
        print("InsightFace installed successfully.")
        return 0

    print("InsightFace failed (on Windows this usually means missing C++ Build Tools).")
    print("Installing DeepFace instead (no Build Tools required)...")
    print()

    if run(["deepface>=0.0.79"]):
        print("DeepFace installed successfully. The app will use DeepFace for face recognition.")
        return 0

    print("DeepFace install also failed. Check the errors above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

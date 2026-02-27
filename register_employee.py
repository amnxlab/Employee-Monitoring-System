#!/usr/bin/env python
"""
Employee Registration Utility

Captures face embeddings and registers new employees in the system.
Updated to work with the one-time face recognition system.

Usage:
    python register_employee.py --id EMP001 --name "John Doe"
    python register_employee.py --list
    python register_employee.py --remove EMP001
    python register_employee.py --verify EMP001
    python register_employee.py --test  # Test face detection without registration

Keys during capture:
    SPACE - Capture sample
    Q or ESC - Cancel/Exit
"""

import argparse
import cv2
import sys
import logging
import numpy as np
from pathlib import Path

import config
from core import Camera, FaceRecognizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def list_employees(face_recognizer: FaceRecognizer):
    """List all registered employees."""
    employees = face_recognizer.database.get_all_employees()
    
    if not employees:
        print("\nNo employees registered.")
        return
    
    print(f"\nRegistered Employees ({len(employees)}):")
    print("-" * 40)
    for emp in employees:
        print(f"  ID: {emp.employee_id:<10} Name: {emp.name}")
    print()


def remove_employee(face_recognizer: FaceRecognizer, employee_id: str):
    """Remove an employee from the database."""
    if face_recognizer.database.remove_employee(employee_id):
        print(f"\nEmployee {employee_id} removed successfully.")
    else:
        print(f"\nEmployee {employee_id} not found.")


def capture_face(camera: Camera, face_recognizer: FaceRecognizer, num_samples: int = 5):
    """
    Capture face samples from camera.
    
    Returns:
        Average embedding from captured samples, or None if failed
    """
    print("\n" + "=" * 50)
    print("FACE CAPTURE")
    print("=" * 50)
    print("\nInstructions:")
    print("  1. Position your face in the center of the frame")
    print("  2. Ensure good lighting on your face")
    print("  3. Look at different angles for better recognition")
    print("  4. Press SPACE to capture when ready")
    print("  5. Press Q or ESC to cancel")
    print("\n" + "-" * 50)
    
    embeddings = []
    sample_count = 0
    
    while sample_count < num_samples:
        frame = camera.read()
        if frame is None:
            continue
        
        h, w = frame.shape[:2]
        scale = min(960 / w, 540 / h, 1.0)
        if scale < 1.0:
            display = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            display = frame.copy()
        
        faces = face_recognizer.detect_faces(frame)
        
        status_color = (0, 0, 255)
        status_text = "No face detected"
        
        if len(faces) == 1:
            face = faces[0]
            x1, y1, x2, y2 = [int(c * scale) for c in face.bbox]
            
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            status_color = (0, 255, 0)
            status_text = f"Face detected (confidence: {face.confidence:.2f})"
            
        elif len(faces) > 1:
            status_text = "Multiple faces detected - ensure only one person"
            for face in faces:
                x1, y1, x2, y2 = [int(c * scale) for c in face.bbox]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
        
        cv2.putText(display, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.putText(display, f"Samples: {sample_count}/{num_samples}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        hint = ["Front", "Left", "Right", "Up", "Down"][sample_count] if sample_count < 5 else "Any"
        cv2.putText(display, f"Next angle: {hint}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        
        cv2.putText(display, "SPACE: Capture | Q/ESC: Cancel", (10, display.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("Employee Registration", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            print("\nRegistration cancelled.")
            cv2.destroyAllWindows()
            return None
        
        elif key == ord(' '):
            if len(faces) == 1:
                embeddings.append(faces[0].embedding)
                sample_count += 1
                print(f"  Sample {sample_count}/{num_samples} captured")
                
                flash = np.ones_like(display) * 255
                cv2.imshow("Employee Registration", flash)
                cv2.waitKey(100)
            else:
                print("  Cannot capture - ensure exactly one face is visible")
    
    cv2.destroyAllWindows()
    
    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        return avg_embedding
    
    return None


def register_employee(
    camera: Camera,
    face_recognizer: FaceRecognizer,
    employee_id: str,
    name: str
):
    """Register a new employee."""
    existing = face_recognizer.database.get_employee(employee_id)
    if existing:
        print(f"\nEmployee {employee_id} already exists as '{existing.name}'")
        response = input("Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("Registration cancelled.")
            return False
    
    print(f"\nRegistering: {name} (ID: {employee_id})")
    
    embedding = capture_face(camera, face_recognizer)
    
    if embedding is None:
        print("\nRegistration failed - no valid face captured.")
        return False
    
    face_recognizer.register_with_embedding(employee_id, name, embedding)
    
    print("\n" + "=" * 50)
    print("REGISTRATION SUCCESSFUL")
    print("=" * 50)
    print(f"  Employee ID: {employee_id}")
    print(f"  Name: {name}")
    print(f"  Embedding stored: {config.EMBEDDINGS_PATH}")
    print()
    
    return True


def verify_registration(
    camera: Camera,
    face_recognizer: FaceRecognizer,
    employee_id: str = None
):
    """Verify a registration by testing recognition."""
    if employee_id:
        print(f"\nVerification mode for {employee_id} - press Q or ESC to exit")
    else:
        print("\nTest mode - press Q or ESC to exit")
    
    while True:
        frame = camera.read()
        if frame is None:
            continue
        
        h, w = frame.shape[:2]
        scale = min(960 / w, 540 / h, 1.0)
        if scale < 1.0:
            display = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            display = frame.copy()
        
        faces = face_recognizer.detect_faces(frame)
        
        for face in faces:
            x1, y1, x2, y2 = [int(c * scale) for c in face.bbox]
            
            matched_id = face_recognizer.identify(face)
            
            if employee_id and matched_id == employee_id:
                color = (0, 255, 0)
                label = f"VERIFIED: {face_recognizer.get_employee_name(matched_id)}"
            elif matched_id:
                color = (0, 255, 0) if not employee_id else (0, 165, 255)
                label = f"Match: {face_recognizer.get_employee_name(matched_id)}"
            else:
                color = (0, 0, 255)
                label = "Unknown"
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(display, "Q/ESC: Exit", (10, display.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("Verification", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
    
    cv2.destroyAllWindows()


def test_detection(camera: Camera, face_recognizer: FaceRecognizer):
    """Test face detection without registration - useful for debugging."""
    print("\nTest mode - testing face detection")
    print("Press Q or ESC to exit")
    
    verify_registration(camera, face_recognizer, employee_id=None)


def main():
    parser = argparse.ArgumentParser(
        description="Employee Registration Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Register new employee:
    python register_employee.py --id EMP001 --name "John Doe"
  
  List all employees:
    python register_employee.py --list
  
  Remove an employee:
    python register_employee.py --remove EMP001
  
  Verify registration:
    python register_employee.py --verify EMP001
  
  Test face detection:
    python register_employee.py --test

Keys:
  SPACE - Capture sample during registration
  Q/ESC - Cancel or exit
        """
    )
    
    parser.add_argument("--id", type=str, help="Employee ID")
    parser.add_argument("--name", type=str, help="Employee name")
    parser.add_argument("--list", action="store_true", help="List all registered employees")
    parser.add_argument("--remove", type=str, metavar="ID", help="Remove an employee")
    parser.add_argument("--verify", type=str, metavar="ID", help="Verify registration")
    parser.add_argument("--test", action="store_true", help="Test face detection without registration")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera index")
    
    args = parser.parse_args()
    
    face_recognizer = FaceRecognizer()
    
    if args.list:
        list_employees(face_recognizer)
        return 0
    
    if args.remove:
        remove_employee(face_recognizer, args.remove)
        return 0
    
    if not face_recognizer.initialize():
        logger.error("Failed to initialize face recognizer")
        print("\nTo fix face recognition, install:")
        print("  python -m pip install deepface tf-keras")
        return 1
    
    camera = Camera(camera_index=args.camera, ptz_enabled=False)
    if not camera.initialize():
        logger.error("Failed to initialize camera")
        return 1
    
    try:
        if args.test:
            test_detection(camera, face_recognizer)
            return 0
        
        if args.verify:
            verify_registration(camera, face_recognizer, args.verify)
            return 0
        
        if args.id and args.name:
            success = register_employee(camera, face_recognizer, args.id, args.name)
            
            if success:
                response = input("Would you like to verify the registration? (y/n): ").strip().lower()
                if response == 'y':
                    verify_registration(camera, face_recognizer, args.id)
            
            return 0 if success else 1
        
        parser.print_help()
        return 1
        
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())

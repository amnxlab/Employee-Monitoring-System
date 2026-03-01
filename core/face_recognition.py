import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    FaceAnalysis = None  # type: ignore
    _INSIGHTFACE_AVAILABLE = False

try:
    from deepface import DeepFace
    _DEEPFACE_AVAILABLE = True
except (ImportError, ValueError) as e:
    DeepFace = None  # type: ignore
    _DEEPFACE_AVAILABLE = False

import config

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Represents a detected face with embedding."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    embedding: np.ndarray
    confidence: float
    landmarks: Optional[np.ndarray] = None
    
    @property
    def center(self) -> Tuple[int, int]:
        return (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2
        )


@dataclass
class Employee:
    """Employee record with face embedding."""
    employee_id: str
    name: str
    embedding: np.ndarray
    
    def to_dict(self) -> dict:
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Employee":
        return cls(
            employee_id=data["employee_id"],
            name=data["name"],
            embedding=data["embedding"]
        )


class EmployeeDatabase:
    """Manages employee face embeddings storage."""
    
    def __init__(self, embeddings_path: Path = None):
        self.embeddings_path = embeddings_path or config.EMBEDDINGS_PATH
        self.employees: Dict[str, Employee] = {}
        self._load()
    
    def _load(self):
        """Load embeddings from disk."""
        if self.embeddings_path.exists():
            try:
                with open(self.embeddings_path, "rb") as f:
                    data = pickle.load(f)
                
                for emp_id, emp_data in data.items():
                    self.employees[emp_id] = Employee.from_dict(emp_data)
                
                logger.info(f"Loaded {len(self.employees)} employees from database")
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
                self.employees = {}
        else:
            logger.info("No existing embeddings database found")
    
    def save(self):
        """Save embeddings to disk."""
        try:
            self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {emp_id: emp.to_dict() for emp_id, emp in self.employees.items()}
            
            with open(self.embeddings_path, "wb") as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved {len(self.employees)} employees to database")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
    
    def add_employee(self, employee_id: str, name: str, embedding: np.ndarray):
        """Add or update an employee."""
        self.employees[employee_id] = Employee(
            employee_id=employee_id,
            name=name,
            embedding=embedding
        )
        self.save()
        logger.info(f"Added employee: {name} ({employee_id})")
    
    def remove_employee(self, employee_id: str) -> bool:
        """Remove an employee from the database."""
        if employee_id in self.employees:
            del self.employees[employee_id]
            self.save()
            logger.info(f"Removed employee: {employee_id}")
            return True
        return False
    
    def get_employee(self, employee_id: str) -> Optional[Employee]:
        """Get employee by ID."""
        return self.employees.get(employee_id)
    
    def get_all_employees(self) -> List[Employee]:
        """Get all registered employees."""
        return list(self.employees.values())
    
    def get_name(self, employee_id: str) -> str:
        """Get employee name by ID."""
        emp = self.employees.get(employee_id)
        return emp.name if emp else "Unknown"


class FaceRecognizer:
    """
    Face recognition system with InsightFace (primary) or DeepFace (fallback).
    Detects faces, extracts embeddings, and matches against employee database.
    """
    
    def __init__(
        self,
        similarity_threshold: float = None,
        detection_size: Tuple[int, int] = None
    ):
        self.similarity_threshold = similarity_threshold or config.FACE_SIMILARITY_THRESHOLD
        self.detection_size = detection_size or config.FACE_DETECTION_SIZE
        
        self.app: Optional[FaceAnalysis] = None
        self.database = EmployeeDatabase()
        self._initialized = False
        self._backend: Optional[str] = None  # "insightface" or "deepface"
    
    def initialize(self) -> bool:
        """Initialize face recognition (InsightFace or DeepFace fallback)."""
        if _INSIGHTFACE_AVAILABLE and FaceAnalysis is not None:
            try:
                logger.info("Initializing InsightFace...")
                self.app = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                self.app.prepare(
                    ctx_id=0,
                    det_size=self.detection_size,
                    det_thresh=0.3,  # lowered from default 0.5 to catch angled/profile faces
                )
                self._initialized = True
                self._backend = "insightface"
                logger.info("InsightFace initialized successfully")
                return True
            except Exception as e:
                logger.warning(f"InsightFace init failed: {e}, trying DeepFace...")
        
        if _DEEPFACE_AVAILABLE and DeepFace is not None:
            try:
                logger.info("Using DeepFace backend (no Build Tools required)...")
                self._initialized = True
                self._backend = "deepface"
                logger.info("DeepFace backend ready")
                return True
            except Exception as e:
                logger.error(f"DeepFace init failed: {e}")
                return False
        
        logger.error(
            "No face recognition backend available. Install one of:\n"
            "  1. pip install deepface   (recommended on Windows, no Build Tools)\n"
            "  2. pip install insightface   (needs Microsoft C++ Build Tools on Windows)\n"
            "     https://visualstudio.microsoft.com/visual-cpp-build-tools/"
        )
        return False
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect all faces in a frame and extract embeddings.
        Automatically resizes large frames to speed up processing.
        """
        if not self._initialized:
            return []
        
        h, w = frame.shape[:2]
        max_dim = getattr(config, "FACE_FRAME_MAX_DIM", 480)
        scale = 1.0
        small = frame
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))

        if self._backend == "insightface" and self.app is not None:
            faces = self._detect_faces_insightface(small)
        elif self._backend == "deepface":
            faces = self._detect_faces_deepface(small)
        else:
            return []

        if scale != 1.0:
            inv = 1.0 / scale
            for f in faces:
                x1, y1, x2, y2 = f.bbox
                f.bbox = (int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv))
        return faces
    
    def _detect_faces_insightface(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces using InsightFace."""
        try:
            faces = self.app.get(frame)
            detections = []
            for face in faces:
                bbox = face.bbox.astype(int)
                detection = FaceDetection(
                    bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    embedding=face.embedding,
                    confidence=float(face.det_score),
                    landmarks=face.landmark_2d_106 if hasattr(face, "landmark_2d_106") else None
                )
                detections.append(detection)
            return detections
        except Exception as e:
            logger.error(f"InsightFace detection error: {e}")
            return []
    
    def _detect_faces_deepface(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces and get embeddings using DeepFace (no C++ Build Tools)."""
        try:
            extracted = DeepFace.represent(
                frame,
                model_name="Facenet",
                detector_backend="ssd",  # ssd handles angled faces much better than opencv
                enforce_detection=False,
                align=True
            )
            if not extracted:
                return []
            if isinstance(extracted, dict):
                extracted = [extracted]
            detections = []
            h_img, w_img = frame.shape[:2]
            for item in extracted:
                if "embedding" not in item:
                    continue
                fa = item.get("facial_area") or {}
                x, y = fa.get("x", 0), fa.get("y", 0)
                w, h = fa.get("w", w_img), fa.get("h", h_img)
                if w <= 0 or h <= 0:
                    w, h = w_img, h_img
                bbox = (int(x), int(y), int(x + w), int(y + h))
                conf = float(item.get("face_confidence", 0.99))
                detections.append(FaceDetection(
                    bbox=bbox,
                    embedding=np.array(item["embedding"], dtype=np.float32),
                    confidence=conf,
                    landmarks=None
                ))
            return detections
        except Exception as e:
            logger.debug(f"DeepFace detection error: {e}")
            return []
    
    def identify(self, face: FaceDetection) -> Optional[str]:
        """
        Identify a face against the employee database.
        
        Args:
            face: FaceDetection with embedding
            
        Returns:
            Employee ID if match found, None otherwise
        """
        if face.embedding is None:
            return None
        
        best_match_id = None
        best_similarity = -1.0
        
        for employee in self.database.get_all_employees():
            similarity = self._cosine_similarity(face.embedding, employee.embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = employee.employee_id
        
        if best_similarity >= self.similarity_threshold:
            name = self.database.get_name(best_match_id)
            logger.info(f"Face MATCHED: {name} (similarity: {best_similarity:.2f}, threshold: {self.similarity_threshold})")
            return best_match_id
        else:
            logger.debug(f"Face NOT matched: best similarity {best_similarity:.2f} < threshold {self.similarity_threshold}")
            return None
    
    def identify_all(self, faces: List[FaceDetection]) -> Dict[int, str]:
        """
        Identify all faces in a list.
        
        Returns:
            Dict mapping face index to employee ID
        """
        results = {}
        for i, face in enumerate(faces):
            emp_id = self.identify(face)
            if emp_id:
                results[i] = emp_id
        return results
    
    def register_employee(
        self,
        employee_id: str,
        name: str,
        frame: np.ndarray
    ) -> bool:
        """
        Register a new employee from a frame containing their face.
        
        Args:
            employee_id: Unique employee identifier
            name: Employee name
            frame: BGR image containing the employee's face
            
        Returns:
            True if registration successful
        """
        faces = self.detect_faces(frame)
        
        if not faces:
            logger.warning("No face detected in registration frame")
            return False
        
        if len(faces) > 1:
            logger.warning("Multiple faces detected, using the largest one")
            faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        
        face = faces[0]
        self.database.add_employee(employee_id, name, face.embedding)
        
        return True
    
    def register_with_embedding(
        self,
        employee_id: str,
        name: str,
        embedding: np.ndarray
    ):
        """Register an employee with a pre-computed embedding."""
        self.database.add_employee(employee_id, name, embedding)
    
    def get_employee_name(self, employee_id: str) -> str:
        """Get employee name by ID."""
        return self.database.get_name(employee_id)
    
    def get_all_employee_ids(self) -> List[str]:
        """Get all registered employee IDs."""
        return list(self.database.employees.keys())
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        a_norm = a / norm_a
        b_norm = b / norm_b
        return float(np.dot(a_norm, b_norm))
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized

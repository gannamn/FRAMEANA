import math
from statistics import median
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp


# =========================================================
# FRAMEANA - STAGE 1
# Stable Face Analysis only
# No 2D overlay, no CAD, no STL
# =========================================================


# Face side landmarks
LEFT_FACE_SIDE = 234
RIGHT_FACE_SIDE = 454

# Eye corner landmarks - fallback only
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Iris landmarks - preferred when refine_landmarks=True
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Nose width candidate pairs.
# We test several pairs because one pair can be too wide depending on face angle/image.
NOSE_WIDTH_CANDIDATE_PAIRS = [
    (98, 327),
    (97, 326),
    (49, 279),
    (64, 294),
    (129, 358),
]


class FaceAnalysisError(Exception):
    """Raised when face analysis cannot be completed."""


mp_face_mesh = mp.solutions.face_mesh


def _landmark_to_point(landmark, image_width: int, image_height: int) -> Tuple[float, float]:
    """Convert normalized MediaPipe landmark to pixel coordinates."""
    return float(landmark.x * image_width), float(landmark.y * image_height)


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance in pixels."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _average_points(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Average a list of pixel points."""
    if not points:
        raise FaceAnalysisError("Cannot average an empty list of points.")

    x = sum(point[0] for point in points) / len(points)
    y = sum(point[1] for point in points) / len(points)
    return x, y


def _has_iris_landmarks(landmarks) -> bool:
    """Check if MediaPipe returned iris landmarks."""
    return len(landmarks) > max(RIGHT_IRIS)


def _get_eye_centers(
    landmarks,
    image_width: int,
    image_height: int,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Get stable eye centers.

    Priority:
    1. Iris center if available.
    2. Eye corner midpoint as fallback.
    """
    if _has_iris_landmarks(landmarks):
        left_eye_points = [
            _landmark_to_point(landmarks[index], image_width, image_height)
            for index in LEFT_IRIS
        ]

        right_eye_points = [
            _landmark_to_point(landmarks[index], image_width, image_height)
            for index in RIGHT_IRIS
        ]

        left_eye_center = _average_points(left_eye_points)
        right_eye_center = _average_points(right_eye_points)

        return left_eye_center, right_eye_center

    left_outer = _landmark_to_point(landmarks[LEFT_EYE_OUTER], image_width, image_height)
    left_inner = _landmark_to_point(landmarks[LEFT_EYE_INNER], image_width, image_height)
    right_inner = _landmark_to_point(landmarks[RIGHT_EYE_INNER], image_width, image_height)
    right_outer = _landmark_to_point(landmarks[RIGHT_EYE_OUTER], image_width, image_height)

    left_eye_center = _average_points([left_outer, left_inner])
    right_eye_center = _average_points([right_inner, right_outer])

    return left_eye_center, right_eye_center


def _get_stable_nose_width(
    landmarks,
    image_width: int,
    image_height: int,
    eye_distance: float,
) -> Tuple[float, Dict[str, object]]:
    """
    Estimate stable nose width.

    The old single pair can return a value that is too large.
    This function calculates multiple candidates and keeps realistic values
    based on the ratio to eye distance.

    A common useful nose-to-eye-distance ratio is usually much lower than 0.69,
    so we filter very wide candidates.
    """
    candidates = []

    for left_index, right_index in NOSE_WIDTH_CANDIDATE_PAIRS:
        left_point = _landmark_to_point(landmarks[left_index], image_width, image_height)
        right_point = _landmark_to_point(landmarks[right_index], image_width, image_height)
        width = _distance(left_point, right_point)

        ratio = width / eye_distance if eye_distance > 0 else 0

        candidates.append(
            {
                "pair": f"{left_index}-{right_index}",
                "width": round(width, 2),
                "ratio_to_eye": round(ratio, 3),
            }
        )

    realistic_widths = [
        item["width"]
        for item in candidates
        if 0.20 <= item["ratio_to_eye"] <= 0.55
    ]

    if realistic_widths:
        selected_width = float(median(realistic_widths))
        method = "median_of_realistic_candidates"
    else:
        selected_width = float(min(item["width"] for item in candidates))
        method = "fallback_min_candidate"

    debug = {
        "nose_width_method": method,
        "nose_candidates": candidates,
    }

    return selected_width, debug


def analyze_face(image_path: str) -> Dict[str, object]:
    """
    Analyze one face image and return facial measurements.

    Returned measurements:
    - face_width
    - eye_distance
    - nose_width
    - head_angle
    - center_x
    - center_y

    Unit: pixels
    """
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise FaceAnalysisError("Could not read image file. Please upload a valid JPG or PNG image.")

    image_height, image_width = image_bgr.shape[:2]

    if image_width <= 0 or image_height <= 0:
        raise FaceAnalysisError("Invalid image size.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        raise FaceAnalysisError("No face detected. Please upload a clear front-facing face image.")

    landmarks = results.multi_face_landmarks[0].landmark

    left_face = _landmark_to_point(landmarks[LEFT_FACE_SIDE], image_width, image_height)
    right_face = _landmark_to_point(landmarks[RIGHT_FACE_SIDE], image_width, image_height)

    left_eye, right_eye = _get_eye_centers(
        landmarks=landmarks,
        image_width=image_width,
        image_height=image_height,
    )

    face_width = _distance(left_face, right_face)
    eye_distance = _distance(left_eye, right_eye)

    nose_width, nose_debug = _get_stable_nose_width(
        landmarks=landmarks,
        image_width=image_width,
        image_height=image_height,
        eye_distance=eye_distance,
    )

    center_x = (left_eye[0] + right_eye[0]) / 2.0
    center_y = (left_eye[1] + right_eye[1]) / 2.0

    head_angle = math.degrees(
        math.atan2(
            right_eye[1] - left_eye[1],
            right_eye[0] - left_eye[0],
        )
    )

    face_ratio_to_eye = face_width / eye_distance if eye_distance > 0 else 0
    nose_ratio_to_eye = nose_width / eye_distance if eye_distance > 0 else 0

    measurements = {
        "face_width": round(face_width, 2),
        "eye_distance": round(eye_distance, 2),
        "nose_width": round(nose_width, 2),
        "head_angle": round(head_angle, 2),
        "center_x": round(center_x, 2),
        "center_y": round(center_y, 2),
    }

    debug = {
        "image_width": image_width,
        "image_height": image_height,
        "face_ratio_to_eye": round(face_ratio_to_eye, 3),
        "nose_ratio_to_eye": round(nose_ratio_to_eye, 3),
        "left_eye": [round(left_eye[0], 2), round(left_eye[1], 2)],
        "right_eye": [round(right_eye[0], 2), round(right_eye[1], 2)],
        **nose_debug,
    }

    return {
        "measurements": measurements,
        "debug": debug,
    }
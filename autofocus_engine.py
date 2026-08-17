"""
Multi-Stage Contrast-Detection Autofocus Engine
------------------------------------------------
Hardware-agnostic core logic of a closed-loop, contrast-based
(Laplacian) autonomous focus algorithm, as used in zoom-lens
electro-optical camera systems.

Core idea:
1) Measure high-frequency (edge) energy in the center region (ROI)
   of the frame to compute a "sharpness score".
2) Sweep motor positions in progressively narrower stages: a wide
   coarse scan first, then a fine scan within the region found by
   the coarse scan, then a micro scan within that. This reaches the
   same precision as a single full-resolution scan with far less
   motor travel.
3) At each stage, instead of picking the single highest-scoring
   position, pick the CENTER of the plateau near the peak (i.e. all
   positions within a threshold of the max score). Near best focus,
   the system sits within the lens's depth of focus, where blur
   spot size stays close to the diffraction limit and barely changes
   with small position shifts — so the sharpness curve is flat near
   the top, not a sharp spike. Locking onto a single sample there is
   fragile against mechanical backlash and sensor noise; locking
   onto the plateau's center is far more repeatable.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SearchStage:
    """Configuration for one stage of the narrowing search."""
    name: str
    span: int      # width of the position range to scan
    steps: int      # number of sample points within that range


class AutofocusEngine:
    """
    Closed-loop, contrast-detection autofocus controller.

    Encapsulates sharpness scoring, plateau-center peak selection, and
    a multi-stage narrowing search plan, so it can be driven by any
    motor/lens backend (the caller is responsible for actually moving
    the motor and capturing frames between stages).

    Example:
        engine = AutofocusEngine(motor_min=0, motor_max=65535)
        for stage in engine.stages:
            positions = engine.positions_for_stage(stage, center)
            scores = [engine.score(capture_frame_at(p)) for p in positions]
            center = engine.find_plateau_center(scores, positions)
        # `center` is now the locked focus position
    """

    def __init__(
        self,
        motor_min: int = 0,
        motor_max: int = 65535,
        roi_radius: int = 40,
        top_n: int = 200,
        plateau_ratio: float = 0.95,
        blur_kernel: tuple[int, int] = (5, 5),
        stages: list[SearchStage] | None = None,
    ):
        self.motor_min = motor_min
        self.motor_max = motor_max
        self.roi_radius = roi_radius
        self.top_n = top_n
        self.plateau_ratio = plateau_ratio
        self.blur_kernel = blur_kernel

        self.stages = stages or [
            SearchStage("coarse", span=motor_max - motor_min, steps=51),
            SearchStage("fine", span=4000, steps=21),
            SearchStage("micro", span=1000, steps=51),
        ]

    def score(self, frame: np.ndarray) -> float:
        """
        Computes a Laplacian-based sharpness score for the center ROI
        of the frame. The score is the mean of the top `top_n`
        highest-frequency responses in the ROI (rather than a full
        variance), which weighs strong edges more heavily and is more
        stable in low-light or low-detail scenes.

        Note: this is a top-N mean, not a variance — despite the
        common "Laplacian variance" naming convention for this
        technique.
        """
        h, w = frame.shape[:2]
        cy, cx = h // 2, w // 2

        # Clip ROI to frame bounds to avoid negative-index wraparound
        y0, y1 = max(0, cy - self.roi_radius), min(h, cy + self.roi_radius)
        x0, x1 = max(0, cx - self.roi_radius), min(w, cx + self.roi_radius)
        roi = frame[y0:y1, x0:x1]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)  # suppress sensor noise
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

        flat = np.sort(laplacian.flatten())
        n = min(self.top_n, flat.size)
        return float(np.mean(flat[-n:])) if n > 0 else 0.0

    def find_plateau_center(self, scores: list[float], positions: list[int]) -> int | None:
        """
        Finds the center of the highest plateau in the score curve.
        Instead of taking the single highest-scoring position, it
        takes the midpoint of all positions within `plateau_ratio` of
        the max score. This reduces false locks caused by mechanical
        jitter or sensor noise near the true peak.
        """
        if not scores:
            return None

        scores_arr = np.array(scores)
        threshold = scores_arr.max() * self.plateau_ratio
        plateau_idx = np.where(scores_arr >= threshold)[0]

        best_idx = (
            plateau_idx[len(plateau_idx) // 2]
            if len(plateau_idx) > 0
            else int(np.argmax(scores_arr))
        )
        return positions[best_idx]

    def positions_for_stage(self, stage: SearchStage, center: int | None = None) -> list[int]:
        """
        Generates the motor positions to sample for a given search
        stage, centered on `center` (the plateau center found by the
        previous stage). If `center` is None, the stage spans the
        full motor range — used for the initial coarse scan.
        """
        if center is None:
            lo, hi = self.motor_min, self.motor_max
        else:
            lo = max(self.motor_min, center - stage.span // 2)
            hi = min(self.motor_max, center + stage.span // 2)
        return [int(p) for p in np.linspace(lo, hi, stage.steps)]

"""
Subject segmentation.

We use MediaPipe Selfie Segmentation by default — it's CPU-friendly, ~0 VRAM,
and ships sharp edges suitable for portrait/subject masking. 

If you later want stronger general-object masks, swap to MobileSAM or
SAM-ViT-B with a single click prompt — the rest of the pipeline doesn't change.
"""
from __future__ import annotations
import numpy as np
import cv2
import mediapipe as mp


class SubjectSegmenter:
    """Returns a soft [0, 1] subject mask the same size as the input."""

    def __init__(self, model_selection: int = 1):
        # model_selection=1 is the "landscape" model — better for full-body
        # and general framing than the close-up portrait model (=0).
        self._mp = mp.solutions.selfie_segmentation
        self._segmenter = self._mp.SelfieSegmentation(model_selection=model_selection)

    def predict(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """
        Args:
            rgb_uint8: HxWx3 uint8 RGB image.
        Returns:
            mask: HxW float32 in [0, 1]. Higher = subject.
        """
        result = self._segmenter.process(rgb_uint8)
        mask = result.segmentation_mask
        if mask is None:
            return np.zeros(rgb_uint8.shape[:2], dtype=np.float32)
        return np.clip(mask.astype(np.float32), 0.0, 1.0)

    @staticmethod
    def sharpen_mask(mask: np.ndarray, threshold: float = 0.5, feather_px: int = 2) -> np.ndarray:
        """
        Convert a soft probability mask into a hard mask with a small feather.
        This is what we'll use to *guide* depth refinement: the binary core
        defines the subject, the feather controls the transition width.
        """
        hard = (mask > threshold).astype(np.uint8) * 255
        # Small morphological cleanup to kill speckle.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, kernel)
        hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, kernel)
        if feather_px > 0:
            hard = cv2.GaussianBlur(hard, (0, 0), sigmaX=feather_px)
        return hard.astype(np.float32) / 255.0
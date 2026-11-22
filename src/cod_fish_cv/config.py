"""Canonical Faster R-CNN & species configuration extracted from legacy notebooks."""

SPECIES_CLASSES = ("1_HYSE", "2_SEI", "3_TORSK", "4_LASK")
ANCHOR_SCALES = (128, 256, 512)
ANCHOR_RATIOS = ((1, 1), (1, 2), (2, 1))
CONVEYOR_ROI_BOUNDS = (0.05, 0.95)

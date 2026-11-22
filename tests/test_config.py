from cod_fish_cv.config import ANCHOR_RATIOS, ANCHOR_SCALES, CONVEYOR_ROI_BOUNDS, SPECIES_CLASSES


def test_species_classes():
    assert SPECIES_CLASSES == ("1_HYSE", "2_SEI", "3_TORSK", "4_LASK")


def test_anchor_config():
    assert ANCHOR_SCALES == (128, 256, 512)
    assert len(ANCHOR_RATIOS) == 3
    assert CONVEYOR_ROI_BOUNDS == (0.05, 0.95)

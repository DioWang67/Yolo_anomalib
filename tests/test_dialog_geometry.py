from PyQt5.QtCore import QSize

from app.gui.dialog_geometry import calculate_dialog_geometry


def test_large_dialog_is_bounded_by_available_screen():
    geometry = calculate_dialog_geometry(
        preferred=QSize(1400, 880),
        minimum=QSize(1100, 720),
        available=QSize(1024, 768),
    )

    assert geometry.initial == QSize(992, 736)
    assert geometry.minimum == QSize(992, 720)


def test_minimum_size_never_exceeds_small_available_screen():
    geometry = calculate_dialog_geometry(
        preferred=QSize(1250, 820),
        minimum=QSize(760, 520),
        available=QSize(400, 300),
    )

    assert geometry.initial == QSize(368, 280)
    assert geometry.minimum == QSize(368, 280)


def test_normal_screen_keeps_preferred_and_requested_minimum():
    geometry = calculate_dialog_geometry(
        preferred=QSize(760, 680),
        minimum=QSize(560, 460),
        available=QSize(1920, 1080),
    )

    assert geometry.initial == QSize(760, 680)
    assert geometry.minimum == QSize(560, 460)

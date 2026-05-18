import os

import cv2
import numpy as np

from core.services.results.crops import save_failure_crops
from core.services.results.image_queue import ImageWriteQueue
from core.services.results.path_manager import ResultPathManager


class DummyLogger:
    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def test_save_failure_crops_writes_reason_named_crops(tmp_path):
    source = np.full((80, 90, 3), 20, dtype=np.uint8)
    source[10:30, 12:34] = 200
    source[35:55, 40:65] = 180
    source[58:75, 5:25] = 160

    queue = ImageWriteQueue(DummyLogger())
    bundle = ResultPathManager(str(tmp_path)).build_paths(
        status="FAIL",
        detector="yolo",
        product="P",
        area="A",
        anomaly_score=None,
    )

    paths = save_failure_crops(
        queue,
        crop_source=source,
        bundle=bundle,
        product="P",
        area="A",
        timestamp_text="123456",
        params=[int(cv2.IMWRITE_PNG_COMPRESSION), 3],
        missing_locations=[
            {
                "class": "nut",
                "expected_key": "nut",
                "bbox": [12, 10, 34, 30],
            }
        ],
        slot_mismatches=[
            {
                "expected_key": "cap",
                "expected_class": "cap",
                "detected_class": "resistor",
                "bbox": [40, 35, 65, 55],
            }
        ],
        detections=[
            {
                "class": "ic",
                "position_expected_key": "ic",
                "position_status": "WRONG",
                "bbox": [5, 58, 25, 75],
            }
        ],
    )
    queue.flush()

    names = [os.path.basename(path) for path in paths]
    assert len(paths) == 3
    assert any("_NG_MISSING_nut_" in name for name in names)
    assert any("_NG_WRONG_COMPONENT_cap_" in name for name in names)
    assert any("_NG_POSITION_SHIFT_ic_" in name for name in names)
    assert all(os.path.exists(path) for path in paths)
    queue.shutdown()

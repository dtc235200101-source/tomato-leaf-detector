from tomato_det.models.base import Detection
from tomato_det.serving.services.recommendations import recommendations_for


def test_recommendations_are_class_specific_and_deduplicated() -> None:
    detections = [
        Detection(3, "late_blight", 0.9, (0.1, 0.1, 0.6, 0.6)),
        Detection(3, "late_blight", 0.8, (0.2, 0.2, 0.7, 0.7)),
        Detection(7, "spider_mites", 0.7, (0.4, 0.4, 0.8, 0.8)),
    ]

    advice = recommendations_for(detections)

    assert [item.class_name for item in advice] == ["late_blight", "spider_mites"]
    assert "lan nhanh" in advice[0].caution
    assert "thiên địch" in advice[1].caution

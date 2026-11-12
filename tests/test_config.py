from pathlib import Path

from tomato_det.common.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_phase_one_configs_have_required_values() -> None:
    data = load_yaml(ROOT / "configs" / "data.yaml")
    evaluation = load_yaml(ROOT / "configs" / "eval.yaml")
    app = load_yaml(ROOT / "configs" / "app.yaml")

    assert data["seed"] == 42
    assert 0 < data["val_fraction"] < 1
    assert data["phash_distance"] >= 0
    assert evaluation["confidence_threshold"] == 0.001
    assert app["host"] == "127.0.0.1"
    assert app["default_model"] == "yolo11s_coco"

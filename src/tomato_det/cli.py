"""Command line interface for the implemented data phase."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from tomato_det.data.inspect import inspect_dataset, prepare_dataset
from tomato_det.eval.artifacts import create_evaluation_lock
from tomato_det.eval.benchmark_latency import benchmark_detector, write_benchmark
from tomato_det.eval.evaluate import evaluate_saved_predictions, evaluate_validation
from tomato_det.eval.report import render_report
from tomato_det.train.train_yolo import load_training_config, run_training, training_preflight

app = typer.Typer(help="Tomato Leaf Detector commands.", no_args_is_help=True)
data_app = typer.Typer(help="Inspect and prepare PlantDoc data.", no_args_is_help=True)
app.add_typer(data_app, name="data")


def _render(payload: dict[str, object]) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


@data_app.command("inspect")
def data_inspect(
    config: Annotated[
        Path | None, typer.Option("--config", exists=False, help="Path to data.yaml")
    ] = None,
) -> None:
    """Report CSV, image, box, and class problems without exporting data."""
    _render(inspect_dataset(config))


@data_app.command("prepare")
def data_prepare(
    config: Annotated[
        Path | None, typer.Option("--config", exists=False, help="Path to data.yaml")
    ] = None,
) -> None:
    """Create leak-checked YOLO/COCO exports; refuses to overwrite an existing export."""
    try:
        _render(prepare_dataset(config))
    except (FileExistsError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error


@app.command("version")
def version() -> None:
    """Print the package version without importing a model or dataset."""
    typer.echo("tomato-leaf-detector 0.1.0")


@app.command("train")
def train(
    experiment: Annotated[
        str, typer.Option("--exp", help="Experiment YAML name without .yaml")
    ] = "yolo11s_coco",
    seed: Annotated[int | None, typer.Option(help="Override the configured seed")] = None,
    device: Annotated[str | None, typer.Option(help="Override the configured device")] = None,
) -> None:
    """Fine-tune YOLO11s only after local data and pretrained weights pass preflight."""
    from tomato_det.common.paths import config_path

    config = load_training_config(config_path(f"experiments/{experiment}.yaml"))
    status = training_preflight(config, requested_device=device)
    if not status.ready:
        _render({"ready": False, "errors": status.errors, "device": asdict(status.device)})
        raise typer.Exit(code=2)
    checkpoint = run_training(config, seed=seed, device=device)
    _render({"ready": True, "checkpoint": checkpoint})


@app.command("eval")
def evaluate(
    experiment: Annotated[
        str, typer.Option("--exp", help="Experiment YAML name without .yaml")
    ] = "yolo11s_coco",
    checkpoint: Annotated[Path, typer.Option(help="Local best.pt from the completed run")] = Path(
        "weights/yolo11s_coco/best.pt"
    ),
    split: Annotated[str, typer.Option(help="Phase 2 permits validation only")] = "val",
) -> None:
    """Run a real validation evaluation; test-set evaluation requires later approval."""
    if split != "val":
        raise typer.BadParameter("Phase 2 only permits --split val")
    from tomato_det.common.paths import config_path, project_root

    config = load_training_config(config_path(f"experiments/{experiment}.yaml"))
    resolved_checkpoint = checkpoint if checkpoint.is_absolute() else project_root() / checkpoint
    result = evaluate_validation(config, resolved_checkpoint)
    _render({"metrics_file": result, "split": "val"})


@app.command("eval-artifacts")
def evaluate_artifacts(
    annotations: Annotated[Path, typer.Option(help="COCO annotation JSON for val or test")],
    predictions: Annotated[Path, typer.Option(help="Real-model prediction JSON in pixel xyxy")],
    split: Annotated[str, typer.Option(help="val, or test after an evaluation lock")] = "val",
    output: Annotated[Path, typer.Option(help="New metrics JSON to write")] = Path(
        "reports/metrics.json"
    ),
    checkpoint: Annotated[Path | None, typer.Option(help="Frozen checkpoint for test only")] = None,
    data_yaml: Annotated[
        Path | None, typer.Option(help="Frozen dataset.yaml for test only")
    ] = None,
    evaluation_lock: Annotated[
        Path | None, typer.Option(help="Lock manifest for test only")
    ] = None,
    operating_threshold: Annotated[
        float | None,
        typer.Option(help="Confidence used for P/R/confusion; default: eval.yaml AP threshold"),
    ] = None,
) -> None:
    """Score persisted real predictions; final TEST rejects an unfrozen configuration."""
    from tomato_det.common.classes import load_classes
    from tomato_det.common.config import load_yaml
    from tomato_det.common.paths import config_path

    settings = load_yaml(config_path("eval.yaml"))
    classes = load_classes(config_path("classes.yaml"))
    result = evaluate_saved_predictions(
        annotations=annotations,
        predictions=predictions,
        class_ids=[item.id for item in classes],
        split=split,
        confidence_threshold=float(settings["confidence_threshold"]),
        operating_iou=float(settings.get("operating_iou", 0.5)),
        operating_confidence=operating_threshold,
        output_path=output,
        checkpoint=checkpoint,
        data_yaml=data_yaml,
        evaluation_lock=evaluation_lock,
    )
    _render({"metrics_file": result, "split": split})


@app.command("predict")
def predict_artifacts(
    experiment: Annotated[
        str, typer.Option("--exp", help="Experiment YAML name without .yaml")
    ] = "yolo11s_coco",
    checkpoint: Annotated[Path, typer.Option(help="Local checkpoint to run")] = Path(
        "weights/yolo11s_coco/best.pt"
    ),
    split: Annotated[str, typer.Option(help="train, val, or test")] = "val",
    output: Annotated[
        Path | None, typer.Option(help="New prediction JSON; defaults to the run directory")
    ] = None,
    conf: Annotated[float, typer.Option(help="Low confidence threshold for AP")] = 0.001,
    iou: Annotated[float, typer.Option(help="NMS IoU threshold")] = 0.6,
    image_size: Annotated[int | None, typer.Option(help="Override experiment image size")] = None,
    max_det: Annotated[int, typer.Option(help="Maximum detections per image")] = 100,
    device: Annotated[str, typer.Option(help="cpu, cuda:0, or auto")] = "auto",
) -> None:
    """Export real predictions in pixel xyxy for the unified evaluator."""
    from tomato_det.common.paths import config_path, project_root
    from tomato_det.eval.predictions import export_predictions
    from tomato_det.train.train_yolo import load_training_config

    config = load_training_config(config_path(f"experiments/{experiment}.yaml"))
    resolved_checkpoint = checkpoint if checkpoint.is_absolute() else project_root() / checkpoint
    destination = output or config.run_dir / f"predictions_{split}.json"
    result = export_predictions(
        checkpoint=resolved_checkpoint,
        split=split,
        output_path=destination,
        confidence_threshold=conf,
        nms_iou=iou,
        image_size=image_size or config.image_size,
        max_detections=max_det,
        device=device,
    )
    _render({"predictions": result, "split": split})


@app.command("choose-threshold")
def choose_threshold(
    annotations: Annotated[Path, typer.Option(help="COCO validation annotation JSON")],
    predictions: Annotated[Path, typer.Option(help="Validation predictions in pixel xyxy")],
    output: Annotated[Path | None, typer.Option(help="Optional JSON artifact to write")] = None,
    iou: Annotated[float, typer.Option(help="Matching IoU used for the operating point")] = 0.5,
) -> None:
    """Choose a deployment confidence threshold using validation data only."""
    from tomato_det.common.classes import load_classes
    from tomato_det.common.paths import config_path
    from tomato_det.eval.artifacts import read_coco_ground_truth, read_predictions
    from tomato_det.eval.operating_point import select_operating_point

    classes = load_classes(config_path("classes.yaml"))
    result = select_operating_point(
        read_coco_ground_truth(annotations),
        read_predictions(predictions),
        class_ids=[item.id for item in classes],
        operating_iou=iou,
    )
    payload = {
        "schema_version": 1,
        "split": "val",
        "annotations": str(annotations),
        "predictions": str(predictions),
        "operating_iou": iou,
        "operating_point": result,
    }
    if output is not None:
        if output.exists():
            raise typer.BadParameter(f"Refusing to overwrite threshold artifact: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _render({"threshold_file": output, "operating_point": result})
        return
    _render(payload)


@app.command("freeze-eval")
def freeze_eval(
    checkpoint: Annotated[Path, typer.Option(help="Chosen checkpoint after validation")],
    data_yaml: Annotated[Path, typer.Option(help="Processed dataset.yaml")],
    validation_metrics: Annotated[Path, typer.Option(help="Existing validation metrics JSON")],
    output: Annotated[Path, typer.Option(help="New immutable test lock JSON")] = Path(
        "runs/evaluation_lock.json"
    ),
) -> None:
    """Freeze a checkpoint/data pair after validation, before the single final TEST run."""
    result = create_evaluation_lock(
        checkpoint=checkpoint,
        data_yaml=data_yaml,
        validation_metrics=validation_metrics,
        output_path=output,
    )
    _render({"evaluation_lock": result})


@app.command("report")
def report(
    metrics: Annotated[Path, typer.Option(help="Measured unified metrics JSON")],
    output_dir: Annotated[Path, typer.Option(help="Directory for generated tables/report")] = Path(
        "reports"
    ),
    figures: Annotated[bool, typer.Option(help="Generate PNG plots (requires matplotlib)")] = True,
    validation_metrics: Annotated[
        Path | None, typer.Option(help="Optional validation metrics JSON")
    ] = None,
    operating_point: Annotated[
        Path | None, typer.Option(help="Optional choose-threshold JSON")
    ] = None,
    latency: Annotated[Path | None, typer.Option(help="Optional bench JSON")] = None,
    dataset_report: Annotated[
        Path | None, typer.Option(help="Optional data/interim/prepare_report.json")
    ] = None,
) -> None:
    """Generate tables, plots, and limitation text from one measured metrics artifact."""
    _render(
        render_report(
            metrics,
            output_dir,
            figures=figures,
            validation_metrics=validation_metrics,
            operating_point=operating_point,
            latency=latency,
            dataset_report=dataset_report,
        )
    )


@app.command("bench")
def bench(
    checkpoint: Annotated[Path, typer.Option(help="Local checkpoint to benchmark")],
    image: Annotated[Path, typer.Option(help="Local representative image; it is not uploaded")],
    output: Annotated[Path, typer.Option(help="New benchmark JSON to write")] = Path(
        "reports/latency.json"
    ),
    device: Annotated[str, typer.Option(help="Device recorded with the measurement")] = "auto",
    image_size: Annotated[int, typer.Option(help="Inference image size used for timing")] = 640,
    warmup_runs: Annotated[int, typer.Option(help="Unrecorded warm-up inferences")] = 5,
    measured_runs: Annotated[int, typer.Option(help="Recorded inferences")] = 30,
) -> None:
    """Measure real local inference-only latency after warm-up; never invent timing values."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise typer.BadParameter("numpy and Pillow are required for benchmark") from error
    from tomato_det.models.yolo_detector import YoloDetector

    if not image.is_file():
        raise typer.BadParameter(f"Image is missing: {image}")
    with Image.open(image) as source:
        image_rgb = np.asarray(source.convert("RGB"))
    detector = YoloDetector(model_id="yolo11s_coco", checkpoint=checkpoint, device=device)
    summary = benchmark_detector(
        detector,
        image_rgb,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        imgsz=image_size,
    )
    result = write_benchmark(
        summary,
        output,
        model_id=detector.model_id,
        device=device,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        input_description=f"{image.name}: {image_rgb.shape[1]}x{image_rgb.shape[0]} RGB",
        image_size=image_size,
    )
    _render({"benchmark_file": result, "latency": summary.as_dict()})


@app.command("serve")
def serve(
    host: Annotated[str | None, typer.Option(help="Bind address; default from app.yaml")] = None,
    port: Annotated[int | None, typer.Option(help="Port; default from app.yaml")] = None,
    reload: Annotated[bool, typer.Option(help="Auto-reload for development")] = False,
) -> None:
    """Run the local FastAPI app on loopback; it never downloads a checkpoint."""
    try:
        import uvicorn
    except ImportError as error:
        raise typer.BadParameter("uvicorn is required to serve the app") from error
    from tomato_det.common.config import load_yaml
    from tomato_det.common.paths import config_path

    config = load_yaml(config_path("app.yaml"))
    bind_host = host or str(config.get("host", "127.0.0.1"))
    bind_port = port or int(config.get("port", 8000))
    uvicorn.run(
        "tomato_det.serving.main:app",
        host=bind_host,
        port=bind_port,
        reload=reload,
        log_level="info",
    )


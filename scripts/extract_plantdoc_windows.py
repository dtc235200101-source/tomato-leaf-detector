"""Extract the PlantDoc tomato subset on Windows without invalid upstream filenames.

The upstream repository has a few names containing ``?``, which Windows cannot
checkout.  This script reads the already-exported CSV files, extracts only image
blobs belonging to configured tomato classes from the Git object database, and
records every skipped path rather than silently changing a filename.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from tomato_det.common.classes import class_lookup, load_classes, normalize_source_label
from tomato_det.common.paths import config_path, project_root

INVALID_WINDOWS_FILENAME = set('<>:"/\\|?*')


def _is_windows_safe(filename: str) -> bool:
    return not any(character in INVALID_WINDOWS_FILENAME for character in filename)


def _git_blob(repository: Path, revision: str, relative_path: str) -> bytes:
    command = [
        "git",
        "-c",
        f"safe.directory={repository.as_posix()}",
        "-C",
        str(repository),
        "show",
        f"{revision}:{relative_path}",
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise FileNotFoundError(f"Cannot read {relative_path} from {revision}: {message}")
    return completed.stdout


def extract_tomato_images(repository: Path, *, revision: str = "FETCH_HEAD") -> dict[str, object]:
    """Extract configured-class images from an upstream Git repository into raw split folders."""
    classes = class_lookup(load_classes(config_path("classes.yaml")))
    report: dict[str, object] = {
        "revision": revision,
        "written": 0,
        "skipped_invalid_filename": [],
        "missing": [],
    }
    for split, folder in (("train", "TRAIN"), ("test", "TEST")):
        csv_path = repository / f"{split}_labels.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing annotation CSV: {csv_path}")
        destination = repository / folder
        destination.mkdir(exist_ok=True)
        filenames: set[str] = set()
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                filename = row.get("filename", "")
                source_class = row.get("class", "")
                if normalize_source_label(source_class) in classes:
                    filenames.add(filename)
        for filename in sorted(filenames):
            source_path = f"{folder}/{filename}"
            if not _is_windows_safe(filename):
                report["skipped_invalid_filename"].append(source_path)
                continue
            target = destination / filename
            if target.is_file():
                continue
            try:
                target.write_bytes(_git_blob(repository, revision, source_path))
            except FileNotFoundError:
                report["missing"].append(source_path)
                continue
            report["written"] = int(report["written"]) + 1
    interim = project_root() / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    output = interim / "windows_extraction_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    raw_repository = project_root() / "data" / "raw" / "plantdoc"
    print(json.dumps(extract_tomato_images(raw_repository), indent=2))

# Reproducible split manifests

`tdet data prepare` writes `train.txt`, `val.txt`, and `test.txt` here. They identify the source
image IDs and are safe to commit only when they contain no private image content or sensitive paths.
Do not edit generated split files by hand; regenerate them from the documented seed and source data.
